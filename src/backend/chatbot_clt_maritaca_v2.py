"""
FASE 3 (Versão V4.3): Agente RAG com Dupla Checagem (Rascunho -> Crítica -> Revisão)
=================================================================================

Este script implementa a arquitetura de Agente V4 (Self-Correction).
O LLM é forçado a um loop de 3 etapas para garantir a precisão:
1.  Decidir Buscar (Loop 1)
2.  Gerar Rascunho (Loop 2)
3.  Criticar e Revisar (Loop 3)

Stack: Maritaca (sabia-3) + BGE-M3 + ChromaDB + BM25
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

#  Precisamos do JSON original para o BM25 e como mapeamento
PATH_MAPPING_FILE = os.path.join(BASE_DIR, 'data', 'processed_data', 'clt_chunks.json') 

# ---  Modelos Maritaca AI ---
LLM_MODEL_NAME = 'sabia-3'
LLM_MODEL_EXPANSION = 'sabia-3' # Usaremos o mesmo modelo para expansão
# ---------------------------------

# ---  Definição dos Prompts do Agente (V4.3) ---

# Prompt 1: O cérebro inicial que decide se deve buscar.
SYSTEM_PROMPT_AGENT = """
Você é um agente jurídico. Sua tarefa é analisar a pergunta do usuário.
- Se for um cumprimento ("Olá", "Obrigado"), responda diretamente.
- Se for uma pergunta técnica sobre leis trabalhistas, você DEVE usar a ferramenta `tool_buscar_na_clt`.
"""

# Prompt 2: O assistente júnior que gera o rascunho.
SYSTEM_PROMPT_DRAFTER = """
Você é um assistente jurídico júnior.
Sua tarefa é gerar um rascunho de resposta para a "PERGUNTA DO USUÁRIO" usando APENAS o "CONTEXTO" fornecido.
Seja direto e cite as fontes (Artigo e Origem) que você usou.
Se o contexto for irrelevante, apenas diga "CONTEXTO IRRELEVANTE".
"""

# Prompt 3: O advogado sênior que critica e revisa o rascunho.
SYSTEM_PROMPT_CRITIC = """
Você é um Advogado Sênior Revisor. Sua tarefa é revisar o "RASCUNHO DE RESPOSTA" gerado por seu assistente júnior.

Você deve seguir 3 REGRAS CRÍTICAS de revisão:

1.  **CHEQUE A PRECISÃO FÁTICA:** O rascunho é 100% suportado pelo "CONTEXTO"? Se o rascunho mencionar um fato (ex: "20 de dezembro") que NÃO está no texto do "CONTEXTO", isso é uma ALUCINAÇÃO e deve ser removido.
2.  **CHEQUE AS CITAÇÕES:** As citações no rascunho (ex: "CLT, Art. X") correspondem exatamente à "Origem" e ao "Artigo" do "CONTEXTO"? Se o rascunho citar "CLT, Art. 2º" mas o contexto mostrar "Origem: Lei 4.090", a citação está ERRADA.
3.  **CHEQUE A GENERALIZAÇÃO:** O rascunho pega uma regra específica (ex: 'para contrato por prazo determinado') e a aplica como se fosse geral? Se sim, isso é um erro grave de generalização.

**Sua Ação:**
-   **Se o Rascunho for bom:** Apenas o aprove e responda ao usuário.
-   **Se o Rascunho tiver erros (Alucinação, Citação Errada, Generalização):** CORRIJA o rascunho e gere a resposta final e precisa.
-   **Se o "CONTEXTO" for irrelevante:** Responda APENAS: "Não encontrei informações sobre este tópico específico na CLT ou nas normas fornecidas para fornecer uma resposta."
"""

# Definição da Ferramenta
TOOL_BUSCAR_CLT = {
    "type": "function",
    "function": {
        "name": "tool_buscar_na_clt",
        "description": "Busca artigos na CLT e normas correlatas para responder a uma pergunta técnica.",
        "parameters": {
            "type": "object",
            "properties": {"pergunta": {"type": "string", "description": "A pergunta original do usuário para a busca RAG-Fusion."}},
            "required": ["pergunta"],
        },
    },
}
AGENT_TOOLS = [TOOL_BUSCAR_CLT]

# ---------------------------------

@st.cache_resource
def load_all_resources():
    logger.info("--- INICIANDO CARREGAMENTO DE RECURSOS (V4.3 - Maritaca) ---")
    load_dotenv(PATH_ENV_FILE)
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords')
    stopwords_pt = set(nltk.corpus.stopwords.words('portuguese'))
    stopwords_pt.update(list(string.punctuation))
    with open(PATH_MAPPING_FILE, 'r', encoding='utf-8') as f:
        chunks_mapeados = json.load(f)
    embedding_model = SentenceTransformer(MODELO_EMBEDDING)
    client = chromadb.PersistentClient(path=PATH_CHROMA_DB, settings=chromadb.config.Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    with open(PATH_BM25_INDEX, 'rb') as f:
        bm25_index = pickle.load(f)
    api_key = os.getenv("OPENAI_API_KEY") # Chave Maritaca
    if not api_key: st.error("Erro: OPENAI_API_KEY (Maritaca) não encontrada."); st.stop()
    maritaca_client = OpenAI(api_key=api_key, base_url="https://chat.maritaca.ai/api")
    logger.info("Cliente Maritaca AI configurado.")
    logger.info("--- TODOS OS RECURSOS FORAM CARREGADOS ---")
    return stopwords_pt, chunks_mapeados, embedding_model, collection, bm25_index, maritaca_client

# --- Funções Internas de Busca (Sem alteração) ---

def preprocessar_query_bm25(texto: str, stopwords: set) -> list:
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = texto.split()
    return [token for token in tokens if token not in stopwords]

def expandir_query_com_llm(query: str, llm_client: OpenAI) -> list:
    logger.info(f"Expandindo query (V4.3/Maritaca): '{query}'")
    system_prompt = """
    Você é um especialista em direito do trabalho brasileiro. Sua tarefa é analisar a pergunta do usuário
    e gerar 3 variações de busca para encontrar os artigos de lei corretos.
    REGRAS IMPORTANTES:
    1.  Traduza termos coloquiais para seus termos jurídicos formais.
    2.  Inclua nomes das leis ou artigos principais, se souber.
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
            model=LLM_MODEL_EXPANSION, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5, max_tokens=256
        )
        novas_queries_str = response.choices[0].message.content
        novas_queries = [q.strip() for q in novas_queries_str.split(';') if q.strip()]
        todas_as_queries = [query] + [q for q in novas_queries if q not in [query]]
        logger.info(f"Queries expandidas (V4.3): {todas_as_queries}")
        return todas_as_queries
    except Exception as e:
        logger.error(f"Erro ao expandir query (Maritaca): {e}. Usando apenas a query original.")
        return [query]

def get_chroma_results(query: str, k: int = 10) -> dict:
    _, _, embed_model, chroma_collection, _, _ = load_all_resources()
    query_com_instrucao = f"Represente esta frase para buscar passagens relevantes: {query}"
    query_vector = embed_model.encode(query_com_instrucao, normalize_embeddings=True).tolist()
    results = chroma_collection.query(query_embeddings=[query_vector], n_results=k)
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
            idx_str = str(idx); score_rrf = 1.0 / (k_rrf + rank)
            fused_scores[idx_str] = fused_scores.get(idx_str, 0.0) + score_rrf
    return fused_scores

def _execute_hybrid_rrf_search(queries: list, k: int = 5) -> tuple[list, str]:
    _, chunks_map, _, _, _, _ = load_all_resources()
    log_busca = "Iniciando busca RAG-Fusion (V4.3)...\n"
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
    for i, chunk in enumerate(chunks_finais):
         idx_str = resultados_finais_idx_str[i]
         log_busca += f"  - Rank {i+1}: {chunk['artigo_numero']} (Score RRF: {fused_hybrid_scores[idx_str]:.6f})\n"
             
    return chunks_finais, log_busca

def formatar_contexto_para_llm(chunks: list) -> str:
    if not chunks: return "Nenhum contexto encontrado."
    contexto_str = ""
    for i, chunk in enumerate(chunks):
        contexto_str += f"[FONTE {i+1}]\nOrigem: {chunk.get('origem', 'N/A')}\n"
        if chunk.get('titulo_clt') != "N/A": contexto_str += f"Título: {chunk.get('titulo_clt')}\n"
        if chunk.get('capitulo_secao') != "N/A": contexto_str += f"Capítulo/Seção: {chunk.get('capitulo_secao')}\n"
        contexto_str += f"Artigo: {chunk.get('artigo_numero')}\nTexto:\n```\n{chunk['conteudo_chunk']}\n```\n\n"
    return contexto_str

def tool_buscar_na_clt(pergunta: str) -> tuple[str, str]:
    logger.info(f"[Agente] Ferramenta 'tool_buscar_na_clt' ativada com a pergunta: '{pergunta}'")
    _, _, _, _, _, llm_client = load_all_resources()
    queries = expandir_query_com_llm(pergunta, llm_client)
    chunks, log_busca = _execute_hybrid_rrf_search(queries)
    contexto_str = formatar_contexto_para_llm(chunks)
    logger.info("[Agente] Busca concluída. Retornando contexto formatado para o LLM.")
    return contexto_str, log_busca

# --- O Cérebro do Agente (V4.3 - Dupla Checagem) ---
def run_agent_loop(query: str, chat_history: list) -> tuple[str, str]:
    """
    Gerencia o loop de 3 etapas (Decidir -> Rascunhar -> Criticar)
    """
    _, _, _, _, _, llm_client = load_all_resources() # Cliente Maritaca
    log_de_busca = "(Nenhuma busca na CLT foi necessária)"
    
    # Prepara o histórico da conversa
    messages = [{"role": "system", "content": SYSTEM_PROMPT_AGENT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": query})

    try:
        # --- PASSO 1: O AGENTE DECIDE SE DEVE USAR A FERRAMENTA ---
        logger.info(f"[Agente] Loop 1 (Decisão): Enviando pergunta ao LLM ({LLM_MODEL_NAME})...")
        response_decisao = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME, messages=messages,
            tools=AGENT_TOOLS, tool_choice="auto", temperature=0.0
        )
        response_message = response_decisao.choices[0].message
        messages.append(response_message)
        
        # --- CASO A: O AGENTE DECIDE USAR A FERRAMENTA ---
        if response_message.tool_calls:
            logger.info("[Agente] Decisão: Usar a ferramenta 'tool_buscar_na_clt'.")
            tool_call = response_message.tool_calls[0]
            
            # Executa a ferramenta (nosso RAG V3.2)
            tool_args = json.loads(tool_call.function.arguments)
            contexto_str, log_de_busca = tool_buscar_na_clt(pergunta=tool_args.get("pergunta"))
            
            # Adiciona o resultado da ferramenta ao histórico
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": contexto_str})
            
            # --- PASSO 2: O AGENTE GERA O RASCUNHO INTERNO ---
            logger.info("[Agente] Loop 2 (Rascunho): Gerando rascunho da resposta...")
            # Adiciona a instrução do Rascunho
            messages_rascunho = [
                {"role": "system", "content": SYSTEM_PROMPT_DRAFTER},
                {"role": "user", "content": f"PERGUNTA DO USUÁRIO:\n{query}\n\nCONTEXTO:\n{contexto_str}"}
            ]
            response_rascunho = llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages_rascunho,
                temperature=0.0
            )
            rascunho = response_rascunho.choices[0].message.content
            logger.info(f"[Agente] Rascunho gerado: {rascunho[:80]}...")
            
            # --- PASSO 3: O AGENTE CRITICA E GERA A RESPOSTA FINAL ---
            logger.info("[Agente] Loop 3 (Crítica/Revisão): Gerando resposta final...")
            messages_critica = [
                {"role": "system", "content": SYSTEM_PROMPT_CRITIC},
                {"role": "user", "content": f"PERGUNTA DO USUÁRIO:\n{query}\n\nCONTEXTO:\n{contexto_str}\n\nRASCUNHO DE RESPOSTA:\n{rascunho}"}
            ]
            
            response_final = llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages_critica,
                temperature=0.0,
                max_tokens=2048
            )
            resposta_corrigida = response_final.choices[0].message.content
            logger.info("[Agente] Resposta final (corrigida) gerada.")
            return resposta_corrigida, log_de_busca

        # --- CASO B: O AGENTE DECIDE RESPONDER DIRETAMENTE ---
        else:
            logger.info("[Agente] Decisão: Responder diretamente (sem ferramentas).")
            return response_message.content, log_de_busca

    except (RateLimitError, BadRequestError) as e:
        logger.error(f"Erro na API (Maritaca/OpenAI): {e}")
        st.error(f"Erro de comunicação com a API: {e.message}")
        return f"❌ Erro ao processar a solicitação (API): {e.message}", log_de_busca
    except Exception as e:
        logger.error(f"Erro inesperado no loop do agente: {e}")
        st.exception(e)
        return f"❌ Erro inesperado no agente: {e}", log_de_busca

# --- Interface Gráfica (Streamlit) ---
def main():
    st.set_page_config(
        page_title="Assistente Jurídico CLT",
        page_icon="⚖️",
        layout="centered"
    )
    
    st.title("⚖️ Agente Trabalhista CLT (V4.3 - Auto-Correção)")
    st.caption("Agente com Maritaca (Sabia-3) e RAG de Dupla Checagem.")

    with st.spinner("Carregando cérebro do agente (V4.3)..."):
        load_all_resources()

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Olá! Sou seu agente especialista na CLT. Como posso ajudar?"
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

            # A nova função de loop orquestra tudo
            resposta, log_busca = run_agent_loop(prompt, history_for_agent)
            
            with st.expander("Ver Raciocínio do Agente (Debug V4.3)"):
                st.text(log_busca)

            message_placeholder.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})

if __name__ == "__main__":
    main()