"""
FASE 3 (Versão V4.4): Agente RAG com HyDE (Hypothetical Document Embedding)
==========================================================================

Este script (V4.4) corrige a falha de recuperação do RAG (vista no teste do 13º)
substituindo o RAG-Fusion por HyDE.

1.  O Agente (LLM) gera uma "resposta hipotética" (alucinada).
2.  Essa resposta (rica em termos jurídicos) é usada para a busca híbrida (RRF).
3.  O Agente (LLM) usa os artigos reais encontrados para gerar a resposta final.

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
LLM_MODEL_EXPANSION = 'sabia-3' 
# ---------------------------------

# --- (NOVO) Definição do Cérebro do Agente (V4.4 - HyDE) ---
SYSTEM_PROMPT_AGENT = """
Você é um Agente Jurídico especialista em Direito do Trabalho brasileiro.

SUA TAREFA:
1.  Analise a PERGUNTA DO USUÁRIO.
2.  **Decida (Raciocine):**
    a) Se a pergunta for simples (cumprimentos como "Olá", "Obrigado", ou perguntas sobre quem você é), responda diretamente.
    b) Se for uma pergunta técnica sobre leis trabalhistas, você DEVE usar a ferramenta `tool_buscar_na_clt`.
3.  **Após usar a ferramenta:**
    a) Você receberá o contexto (Artigos da CLT).
    b) **Valide o Contexto:** O contexto recebido é relevante para a pergunta?
    c) **Se for relevante:** Responda à pergunta baseando-se EXCLUSIVAMENTE nesse contexto. Cite a fonte (Origem e Artigo). Não generalize regras específicas (ex: "contrato determinado").
    d) **Se o contexto for irrelevante ou vazio:** Responda APENAS: "Não encontrei informações sobre este tópico específico na CLT ou nas normas fornecidas para fornecer uma resposta."
4.  Nunca use seu conhecimento prévio sobre leis, exceto para gerar a busca da ferramenta. Sua resposta final deve ser 100% baseada no contexto fornecido pela ferramenta.
"""

# Definição da Ferramenta
TOOL_BUSCAR_CLT = {
    "type": "function",
    "function": {
        "name": "tool_buscar_na_clt",
        "description": "Busca artigos na CLT e normas correlatas para responder a uma pergunta técnica.",
        "parameters": {
            "type": "object",
            "properties": {"pergunta": {"type": "string", "description": "A pergunta original do usuário."}},
            "required": ["pergunta"],
        },
    },
}
AGENT_TOOLS = [TOOL_BUSCAR_CLT]
# ---------------------------------

@st.cache_resource
def load_all_resources():
    logger.info("--- INICIANDO CARREGAMENTO DE RECURSOS (V4.4 - HyDE) ---")
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

# --- Funções Internas de Busca ---

def preprocessar_query_bm25(texto: str, stopwords: set) -> list:
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = texto.split()
    return [token for token in tokens if token not in stopwords]

# --- (NOVA FUNÇÃO HyDE) ---
def gerar_documento_hipotetico(query: str, llm_client: OpenAI) -> str:
    """Usa o LLM para gerar uma resposta hipotética (alucinada) para a query."""
    logger.info(f"Gerando documento hipotético (HyDE) para: '{query}'")
    
    system_prompt = """
    Você é um especialista em direito do trabalho brasileiro. 
    Responda à pergunta do usuário da forma mais completa e técnica possível, 
    como se fosse um artigo de lei ou a explicação de um advogado.
    Cite artigos e leis fictícias se necessário. 
    Esta resposta será usada para BUSCAR, não será mostrada ao usuário.
    """
    
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_HYDE, # sabia-3
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.5,
            max_tokens=256
        )
        doc_hipotetico = response.choices[0].message.content
        logger.info(f"Documento Hipotético gerado: {doc_hipotetico[:100]}...")
        return doc_hipotetico
    except Exception as e:
        logger.error(f"Erro ao gerar documento HyDE: {e}. Usando query original.")
        return query # Se falhar, usa a própria query

# --- Funções de Busca (Chroma/BM25) ---
def get_chroma_results(query_embedding: np.ndarray, k: int = 10) -> dict:
    """Busca no ChromaDB."""
    _, _, _, chroma_collection, _, _ = load_all_resources()
    query_vector_list = query_embedding.tolist()
    results = chroma_collection.query(query_embeddings=[query_vector_list], n_results=k)
    return {str(idx): i for i, idx in enumerate(results['ids'][0])}

def get_bm25_results(query: str, k: int = 10) -> dict:
    """Busca no BM25."""
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

# --- (ATUALIZADA) Função da Ferramenta (HyDE + RRF) ---
def tool_buscar_na_clt(pergunta: str) -> tuple[str, str]:
    """
    Função wrapper que executa o pipeline V4.4 (HyDE + Híbrido RRF).
    """
    logger.info(f"[Agente] Ferramenta 'tool_buscar_na_clt' ativada com a pergunta: '{pergunta}'")
    _, chunks_map, embed_model, _, _, llm_client = load_all_resources()
    log_busca = ""
    
    # --- PASSO 1: HyDE ---
    # Gera a resposta hipotética (rica em termos jurídicos)
    doc_hipotetico = gerar_documento_hipotetico(pergunta, llm_client)
    log_busca += f"Iniciando busca HyDE (V4.4)...\n"
    log_busca += f"Documento Hipotético (Query de Busca): {doc_hipotetico}\n\n"
    
    # --- PASSO 2: Busca Híbrida (RRF) ---
    # Usamos o documento hipotético para a busca
    
    # A) Busca Vetorial (Chroma)
    # (Instrução BGE-M3 + Documento Hipotético)
    query_com_instrucao = f"Represente esta frase para buscar passagens relevantes: {doc_hipotetico}"
    query_embedding = embed_model.encode(query_com_instrucao, normalize_embeddings=True)
    chroma_ranks = get_chroma_results(query_embedding)
    
    # B) Busca Keyword (BM25)
    # (Apenas o Documento Hipotético)
    bm25_ranks = get_bm25_results(doc_hipotetico)

    # C) Fusão (RRF)
    final_chroma_ranks = {idx: i for i, (idx, score) in enumerate(sorted(chroma_ranks.items(), key=lambda item: item[1]))} # Chroma já é rank
    final_bm25_ranks = {str(idx): i for i, (idx, score) in enumerate(sorted(bm25_ranks.items(), key=lambda item: item[1]))} # BM25 já é rank
    
    todos_indices = set(final_chroma_ranks.keys()).union(set(final_bm25_ranks.keys()))
    fused_hybrid_scores = {}
    
    for idx_str in todos_indices:
        score = (1.0 / (60 + final_chroma_ranks[idx_str])) if idx_str in final_chroma_ranks else 0
        score += (1.0 / (60 + final_bm25_ranks[idx_str])) if idx_str in final_bm25_ranks else 0
        fused_hybrid_scores[idx_str] = score

    resultados_finais_idx_str = sorted(fused_hybrid_scores.keys(), key=lambda idx: fused_hybrid_scores[idx], reverse=True)
    
    # Pega os 5 melhores
    k = 5 
    chunks_finais = [chunks_map[int(idx_str)] for idx_str in resultados_finais_idx_str[:k]]
    
    log_busca += "\n--- Contexto Final Selecionado (Pós-HyDE) ---\n"
    if not chunks_finais:
        log_busca += "Nenhum artigo relevante encontrado."
    else:
        for i, chunk in enumerate(chunks_finais):
             idx_str = resultados_finais_idx_str[i]
             log_busca += f"  - Rank {i+1}: {chunk['artigo_numero']} (Score RRF: {fused_hybrid_scores[idx_str]:.6f})\n"

    # --- PASSO 3: Formatar Contexto ---
    contexto_str = formatar_contexto_para_llm_hyDE(chunks_finais) # Usa a função de formatação
    
    logger.info("[Agente] Busca HyDE concluída. Retornando contexto formatado para o LLM.")
    return contexto_str, log_busca

def formatar_contexto_para_llm_hyDE(chunks: list) -> str:
    """Função de formatação de contexto (extraída da V4.2)."""
    if not chunks: return "Nenhum contexto encontrado."
    contexto_str = "--- CONTEXTO JURÍDICO (CLT E NORMAS CORRELATAS) ---\n\n"
    for i, chunk_data in enumerate(chunks):
        contexto_str += f"[FONTE {i+1}]\nOrigem: {chunk_data.get('origem', 'N/A')}\n"
        if chunk_data.get('titulo_clt', 'N/A') != "N/A": contexto_str += f"Título: {chunk_data.get('titulo_clt')}\n"
        if chunk_data.get('capitulo_secao', 'N/A') != "N/A": contexto_str += f"Capítulo/Seção: {chunk_data.get('capitulo_secao')}\n"
        contexto_str += f"Artigo: {chunk_data.get('artigo_numero')}\nTexto:\n```\n{chunk_data['conteudo_chunk']}\n```\n\n"
    return contexto_str

# --- (ATUALIZADO) O Cérebro do Agente (V4.4 - HyDE) ---
def run_agent_loop(query: str, chat_history: list) -> tuple[str, str]:
    """
    Gerencia o loop ReAct (Reason-Act) do agente usando Maritaca (HyDE).
    """
    _, _, _, _, _, llm_client = load_all_resources() # Cliente Maritaca
    log_de_busca = "(Nenhuma busca na CLT foi necessária)"
    
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
            if tool_call.function.name == "tool_buscar_na_clt":
                tool_args = json.loads(tool_call.function.arguments)
                
                # (MUDANÇA V4.4) Executa a ferramenta HyDE
                contexto_str, log_de_busca = tool_buscar_na_clt(pergunta=tool_args.get("pergunta"))
                
                # Adiciona o resultado da ferramenta ao histórico
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": contexto_str,
                })
                
                # --- PASSO 2: O AGENTE GERA A RESPOSTA FINAL ---
                # (Removemos o loop de "Crítica" - O HyDE deve ter melhorado o contexto)
                logger.info("[Agente] Loop 2 (Geração Final): Enviando contexto para resposta...")
                
                final_response = llm_client.chat.completions.create(
                    model=LLM_MODEL_NAME, # sabia-3
                    messages=messages, # O histórico agora contém a chamada da ferramenta e o contexto
                    temperature=0.0,
                    max_tokens=2048
                )
                return final_response.choices[0].message.content, log_de_busca
            
        # --- CASO B: O AGENTE DECIDE RESPONDER DIRETAMENTE ---
        logger.info("[Agente] Decisão: Responder diretamente.")
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
    
    st.title("⚖️ Agente Trabalhista CLT (V4.4 - HyDE)")
    st.caption("Agente com Maritaca (Sabia-3) e busca Híbrida/HyDE.")

    with st.spinner("Carregando cérebro do agente (V4.4)..."):
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

            resposta, log_busca = run_agent_loop(prompt, history_for_agent)
            
            with st.expander("Ver Raciocínio do Agente (Debug V4.4)"):
                st.text(log_busca)

            message_placeholder.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})

if __name__ == "__main__":
    main()