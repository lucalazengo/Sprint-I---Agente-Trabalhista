"""
FASE 3 (Versão V4.2): Agente RAG com Maritaca AI (Sabia-3)
=========================================================

Este script implementa a arquitetura de Agente V3.3 (Cérebro Crítico),
com a stack de modelos atualizada para Maritaca AI:
-   LLM (Agente): Maritaca (sabia-3)
-   Embedding: BAAI/bge-m3
-   Vector DB: ChromaDB
-   Keyword DB: BM25
-   RAG: RAG-Fusion + RRF
"""

import os
import pickle
import json
import re
import string
import chromadb # (NOVO) Importa Chroma
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI, RateLimitError, BadRequestError
from dotenv import load_dotenv
import streamlit as st
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuração de Caminhos  ---
BASE_DIR = r'/Users/universo369/Documents/UNIVERSO_369/U369_root/GARDEN SOLUTION/adicao_contabilidade/Sprint-I---Agente-Trabalhista'
MODELO_EMBEDDING = 'BAAI/bge-m3'

#  Caminhos dos artefatos
PATH_BM25_INDEX = os.path.join(BASE_DIR, 'data', 'processed_data','indices', 'clt_bm25_bge-m3.pkl')
PATH_CHROMA_DB = os.path.join(BASE_DIR, 'data', 'processed_data','indices', 'chroma_db_bge_m3')
CHROMA_COLLECTION_NAME = "clt_bge_m3"
PATH_ENV_FILE = os.path.join(BASE_DIR, 'src','backend' , 'secrets', '.env')

PATH_MAPPING_FILE = os.path.join(BASE_DIR, 'data', 'processed_data', 'clt_chunks.json') 

# --- Modelos Maritaca AI ---
LLM_MODEL_NAME = 'sabia-3'
LLM_MODEL_EXPANSION = 'sabia-3' # Usaremos o mesmo modelo para expansão
# ---------------------------------

# --- Definição do Cérebro do Agente  ---
SYSTEM_PROMPT_AGENT = """
Você é um Agente Jurídico especialista em Direito do Trabalho brasileiro.
Sua tarefa é seguir um processo rigoroso de raciocínio para responder perguntas.

### PROCESSO DE RACIOCÍNIO OBRIGATÓRIO:

**PASSO 1: ANÁLISE DA PERGUNTA**
-   Analise a PERGUNTA DO USUÁRIO.
-   Se for um cumprimento ("Olá", "Obrigado") ou uma pergunta sobre você, responda diretamente.
-   Se for uma pergunta técnica sobre leis trabalhistas, ATIVE O PASSO 2.

**PASSO 2: DECISÃO DE FERRAMENTA**
-   Seu único conhecimento jurídico vem da base de dados.
-   Sua decisão deve ser SEMPRE usar a ferramenta `tool_buscar_na_clt` para encontrar os artigos relevantes.

**PASSO 3: ANÁLISE CRÍTICA DO CONTEXTO (O MAIS IMPORTANTE)**
-   Você receberá o resultado da ferramenta (os artigos de lei).
-   **CRITIQUE OS RESULTADOS:**
    1.  **Relevância:** Os artigos encontrados respondem DIRETAMENTE à pergunta do usuário?
    2.  **Contexto Específico (Caput):** A regra se aplica a todos ou a um caso específico? (Ex: A regra é só para "contrato por prazo determinado"? [como o Art. 479] É só para "bancários"? [como o Art. 224] É só para "menor"? [como o Art. 402]).
    3.  **Fonte Correta:** Os artigos vieram da "CLT" ou de uma "Lei" correlata? (Ex: Lei nº 4.090/1962).
-   **Se o contexto for insuficiente, irrelevante ou você não tiver certeza:** Responda APENAS: "Não encontrei informações sobre este tópico específico na CLT ou nas normas fornecidas para fornecer uma resposta."

**PASSO 4: GERAÇÃO DA RESPOSTA**
-   Se, e somente se, o PASSO 3 for um sucesso:
-   Responda de forma objetiva, técnica e clara.
-   Baseie-se EXCLUSIVAMENTE nos artigos fornecidos.
-   **CITAÇÃO RIGOROSA:** Ao citar, você DEVE usar a fonte exata do contexto (o campo `Origem`).
    -   Exemplo Correto (CLT): "Conforme a CLT, Art. 467..."
    -   Exemplo Correto (Norma): "Conforme a Lei nº 4.090/1962, Art. 3º..."
-   **NÃO GENERALIZE:** Se a regra for específica (ex: Art. 479), você DEVE dizer: "No caso de contratos por prazo determinado, a CLT, Art. 479, estabelece que..."
"""

# --- Definição da Ferramenta do Agente (V3) ---
TOOL_BUSCAR_CLT = {
    "type": "function",
    "function": {
        "name": "tool_buscar_na_clt",
        "description": "Busca artigos na Consolidação das Leis do Trabalho (CLT) e normas correlatas para responder a uma pergunta técnica sobre direito trabalhista.",
        "parameters": {
            "type": "object",
            "properties": {
                "pergunta": {
                    "type": "string",
                    "description": "A pergunta original do usuário, que será usada para expandir a busca (RAG-Fusion) e encontrar os artigos relevantes.",
                }
            },
            "required": ["pergunta"],
        },
    },
}
AGENT_TOOLS = [TOOL_BUSCAR_CLT]

# ---------------------------------

@st.cache_resource
def load_all_resources():
    logger.info("--- INICIANDO CARREGAMENTO DE RECURSOS (V4.2 - Maritaca) ---")
    
    # 1. Carregar .env
    logger.info(f"Carregando .env de {PATH_ENV_FILE}...")
    load_dotenv(PATH_ENV_FILE)

    # 2. Carregar Stopwords
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords')
    stopwords_pt = set(nltk.corpus.stopwords.words('portuguese'))
    stopwords_pt.update(list(string.punctuation))
    logger.info("Stopwords carregadas.")

    # 3. Carregar Mapeamento de Chunks (para BM25 e metadados)
    logger.info(f"Carregando chunks de {PATH_MAPPING_FILE}...")
    with open(PATH_MAPPING_FILE, 'r', encoding='utf-8') as f:
        chunks_mapeados = json.load(f)
    logger.info(f"{len(chunks_mapeados)} chunks carregados.")
    
    # 4. Carregar Modelo de Embedding (BGE-M3)
    logger.info(f"Carregando modelo de embedding: {MODELO_EMBEDDING}...")
    embedding_model = SentenceTransformer(MODELO_EMBEDDING)
    logger.info("Modelo BGE-M3 carregado.")

    # 5. Carregar Índice ChromaDB
    logger.info(f"Conectando ao ChromaDB em {PATH_CHROMA_DB}...")
    client = chromadb.PersistentClient(
        path=PATH_CHROMA_DB,
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    logger.info(f"Conectado ao ChromaDB. Coleção '{CHROMA_COLLECTION_NAME}' carregada com {collection.count()} itens.")

    # 6. Carregar Índice BM25
    logger.info(f"Carregando índice BM25 de {PATH_BM25_INDEX}...")
    with open(PATH_BM25_INDEX, 'rb') as f:
        bm25_index = pickle.load(f)
    logger.info("Índice BM25 (BGE-M3) carregado.")

    api_key = os.getenv("OPENAI_API_KEY") 
    if not api_key:
        st.error("Erro: OPENAI_API_KEY (contendo a chave Maritaca) não encontrada no seu arquivo .env.")
        st.stop()
    
    maritaca_client = OpenAI(
        api_key=api_key,
        base_url="https://chat.maritaca.ai/api"
    )
    logger.info("Cliente Maritaca AI configurado (via endpoint OpenAI).")
    
    logger.info("--- TODOS OS RECURSOS FORAM CARREGADOS ---")
    
    # Retorna o cliente Maritaca
    return stopwords_pt, chunks_mapeados, embedding_model, collection, bm25_index, maritaca_client

# --- Funções Internas de Busca ---

def preprocessar_query_bm25(texto: str, stopwords: set) -> list:
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = texto.split()
    return [token for token in tokens if token not in stopwords]

def expandir_query_com_llm(query: str, llm_client: OpenAI) -> list:
    """Usa Maritaca (sabia-3) para gerar variações da pergunta."""
    logger.info(f"Expandindo query (V4.2/Maritaca): '{query}'")
    
    system_prompt = """
    Você é um especialista em direito do trabalho brasileiro. Sua tarefa é analisar a pergunta do usuário
    e gerar 3 variações de busca para encontrar os artigos de lei corretos.

    REGRAS IMPORTANTES:
    1.  Traduza termos coloquiais (linguagem do dia-a-dia) para seus termos jurídicos formais.
    2.  Inclua os nomes das leis ou artigos principais, se souber.
    3.  Retorne APENAS as 3 novas perguntas, separadas por ponto e vírgula (;). Sem cabeçalhos ou numeração.

    EXEMPLOS:
    -   Pergunta original: Posso vender parte das férias?
        Resultado: abono pecuniário CLT; converter 1/3 das férias em dinheiro; Art. 143 CLT
    -   Pergunta original: Quando devo pagar o 13º salário?
        Resultado: data limite pagamento gratificação de natal; prazo Lei 4749 13º salário; pagamento primeira parcela décimo terceiro
    -   Pergunta original: Fui demitido, o que eu recebo?
        Resultado: verbas rescisórias demissão sem justa causa; direitos rescisão contrato de trabalho CLT Art 477; multa 40% FGTS
    """
    
    user_prompt = f"Pergunta original: {query}"
    
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_EXPANSION, # sabia-3
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=256
        )
        
        novas_queries_str = response.choices[0].message.content
        novas_queries = [q.strip() for q in novas_queries_str.split(';') if q.strip()]
        
        todas_as_queries = [query]
        for q in novas_queries:
            if q not in todas_as_queries:
                todas_as_queries.append(q)
        
        logger.info(f"Queries expandidas (V4.2): {todas_as_queries}")
        return todas_as_queries
        
    except Exception as e:
        logger.error(f"Erro ao expandir query (Maritaca): {e}. Usando apenas a query original.")
        return [query]

def get_chroma_results(query: str, k: int = 10) -> dict:
    _, _, embed_model, chroma_collection, _, _ = load_all_resources()
    query_com_instrucao = f"Represente esta frase para buscar passagens relevantes: {query}"
    query_vector = embed_model.encode(query_com_instrucao, normalize_embeddings=True)
    query_vector_list = query_vector.tolist()
    results = chroma_collection.query(query_embeddings=[query_vector_list], n_results=k)
    return {str(idx): i for i, idx in enumerate(results['ids'][0])}

def get_bm25_results(query: str, k: int = 10) -> dict:
    stopwords, _, _, _, bm25_idx, _ = load_all_resources()
    query_bm25 = preprocessar_query_bm25(query, stopwords)
    scores_bm25 = bm25_idx.get_scores(query_bm25)
    indices_bm25 = np.argsort(scores_bm25)[::-1][:k]
    return {int(idx): i for i, idx in enumerate(indices_bm25) if scores_bm25[idx] > 0}

def fuse_rrf_results(list_of_rankings: list, k_rrf: int = 60) -> dict:
    fused_scores = {}
    for ranking_dict in list_of_rankings:
        for idx, rank in ranking_dict.items():
            idx_str = str(idx) 
            score_rrf = 1.0 / (k_rrf + rank)
            fused_scores[idx_str] = fused_scores.get(idx_str, 0.0) + score_rrf
    return fused_scores

def _execute_hybrid_rrf_search(queries: list, k: int = 5) -> tuple[list, str]:
    _, chunks_map, _, _, _, llm_client = load_all_resources()
    log_busca = "Iniciando busca RAG-Fusion ...\n"
    log_busca += f"Queries Geradas: {queries}\n\n"
    
    todos_ranks_chroma = [get_chroma_results(q) for q in queries]
    todos_ranks_bm25 = [get_bm25_results(q) for q in queries]

    scores_chroma_fused = fuse_rrf_results(todos_ranks_chroma)
    scores_bm25_fused = fuse_rrf_results(todos_ranks_bm25)
    
    final_chroma_ranks = {idx: i for i, (idx, score) in enumerate(sorted(scores_chroma_fused.items(), key=lambda item: item[1], reverse=True))}
    final_bm25_ranks = {str(idx): i for i, (idx, score) in enumerate(sorted(scores_bm25_fused.items(), key=lambda item: item[1], reverse=True))}
    
    todos_indices = set(final_chroma_ranks.keys()).union(set(final_bm25_ranks.keys()))
    fused_hybrid_scores = {}
    
    for idx_str in todos_indices:
        score = (1.0 / (60 + final_chroma_ranks[idx_str])) if idx_str in final_chroma_ranks else 0
        score += (1.0 / (60 + final_bm25_ranks[idx_str])) if idx_str in final_bm25_ranks else 0
        fused_hybrid_scores[idx_str] = score

    resultados_finais_idx_str = sorted(fused_hybrid_scores.keys(), key=lambda idx: fused_hybrid_scores[idx], reverse=True)
    chunks_finais = [chunks_map[int(idx_str)] for idx_str in resultados_finais_idx_str[:k]]
    
    log_busca += "\n--- Contexto Final Selecionado (Pós-RAG-Fusion) ---\n"
    if not chunks_finais:
        log_busca += "Nenhum artigo relevante encontrado."
    else:
        for i, chunk in enumerate(chunks_finais):
             idx_str = resultados_finais_idx_str[i]
             log_busca += f"  - Rank {i+1}: {chunk['artigo_numero']} (Score RRF: {fused_hybrid_scores[idx_str]:.6f})\n"
             
    return chunks_finais, log_busca

def formatar_contexto_para_llm(chunks: list) -> str:
    if not chunks: return "Nenhum contexto encontrado."
    contexto_str = "--- CONTEXTO JURÍDICO (Leis de Trabalho brasileiras) ---\n\n"
    for i, chunk in enumerate(chunks):
        contexto_str += f"[FONTE {i+1}]\nOrigem: {chunk.get('origem', 'N/A')}\n" # Atualizado para metadados do Chroma
        if chunk.get('titulo_clt') != "N/A": contexto_str += f"Título: {chunk.get('titulo_clt')}\n"
        if chunk.get('capitulo_secao') != "N/A": contexto_str += f"Capítulo/Seção: {chunk.get('capitulo_secao')}\n"
        contexto_str += f"Artigo: {chunk.get('artigo')}\nTexto:\n```\n{chunk['document']}\n```\n\n" # 'document' é a chave do Chroma
    return contexto_str

def tool_buscar_na_clt(pergunta: str) -> tuple[str, str]:
    """Wrapper da ferramenta de busca (V4.2)."""
    logger.info(f"[Agente] Ferramenta 'tool_buscar_na_clt' ativada com a pergunta: '{pergunta}'")
    _, _, _, _, _, llm_client = load_all_resources()
    
    queries = expandir_query_com_llm(pergunta, llm_client)
    chunks_info, log_busca = _execute_hybrid_rrf_search(queries) # Esta função retorna chunks do JSON
    

    
    # Constrói o contexto com base nos chunks_map
    if not chunks_info: 
        return "Nenhum contexto encontrado.", log_busca
        
    contexto_str = "--- CONTEXTO JURÍDICO (Leis de Trabalho brasileiras) ---\n\n"
    for i, chunk_data in enumerate(chunks_info):    
        contexto_str += f"[FONTE {i+1}]\nOrigem: {chunk_data.get('origem', 'N/A')}\n"
        if chunk_data.get('titulo_clt') != "N/A": contexto_str += f"Título: {chunk_data.get('titulo_clt')}\n"
        if chunk_data.get('capitulo_secao') != "N/A": contexto_str += f"Capítulo/Seção: {chunk_data.get('capitulo_secao')}\n"
        contexto_str += f"Artigo: {chunk_data.get('artigo_numero')}\nTexto:\n```\n{chunk_data['conteudo_chunk']}\n```\n\n"

    logger.info("[Agente] Busca concluída. Retornando contexto formatado para o LLM.")
    return contexto_str, log_busca

# ---  O Cérebro do Agente (V4.2 - Maritaca) ---
def run_agent_loop(query: str, chat_history: list) -> tuple[str, str]:
    """Gerencia o loop ReAct (Reason-Act) do agente usando Maritaca."""
    _, _, _, _, _, llm_client = load_all_resources() # Agora é o cliente Maritaca
    log_de_busca = ""
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT_AGENT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": query})

    try:
        logger.info(f"[Agente] Loop 1: Enviando pergunta ao LLM ({LLM_MODEL_NAME}) para decisão...")
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME, # sabia-3
            messages=messages,
            tools=AGENT_TOOLS,
            tool_choice="auto",
            temperature=0.0
        )
        response_message = response.choices[0].message
        messages.append(response_message)
        
        if response_message.tool_calls:
            logger.info("[Agente] O LLM decidiu usar uma ferramenta.")
            tool_call = response_message.tool_calls[0]
            if tool_call.function.name == "tool_buscar_na_clt":
                tool_args = json.loads(tool_call.function.arguments)
                contexto_str, log_de_busca = tool_buscar_na_clt(pergunta=tool_args.get("pergunta"))
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": contexto_str,
                })
                
                logger.info("[Agente] Loop 2: Enviando contexto da ferramenta para geração final...")
                final_response = llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME, # sabia-3
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048
                )
                return final_response.choices[0].message.content, log_de_busca
            
        logger.info("[Agente] O LLM decidiu responder diretamente.")
        return response_message.content, "(Nenhuma busca na CLT foi necessária)"

    except (RateLimitError, BadRequestError) as e:
        logger.error(f"Erro na API (Maritaca/OpenAI): {e}")
        st.error(f"Erro de comunicação com a API: {e.message}")
        return f"❌ Erro ao processar a solicitação (API): {e.message}", log_de_busca
    except Exception as e:
        logger.error(f"Erro inesperado no loop do agente: {e}")
        st.exception(e)
        return f"❌ Erro inesperado no agente: {e}", log_de_busca

# --- Interface Gráfica  ---
def main():
    st.set_page_config(
        page_title="Agente Trabalhista",
        page_icon="⚖️",
        layout="centered"
    )
    
    st.title("Agente Trabalhista")
    st.caption("Agente com Sabia-3 e busca híbrida (BGE-M3 + BM25) em ChromaDB.")

    with st.spinner("Carregando cérebro do agente (V4.2)..."):
        load_all_resources()

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Olá! Sou seu agente especialista em Leis de Trabalho brasileiras. Como posso ajudar?"
        }]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Digite sua dúvida..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Pensando... 🤔")
            
            history_for_agent = [
                msg for msg in st.session_state.messages 
                if msg["role"] in ("user", "assistant")
            ]
            history_for_agent = history_for_agent[:-1] 

            resposta, log_busca = run_agent_loop(prompt, history_for_agent)
            
            with st.expander("Ver Raciocínio do Agente (Debug V4.2)"):
                st.text(log_busca)

            message_placeholder.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})

if __name__ == "__main__":
    main()