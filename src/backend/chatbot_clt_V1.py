"""
FASE 3 (Versão V3): Agente RAG Híbrido (Self-RAG)
=================================================

Este script implementa a arquitetura V3 (Agente).
O LLM (Groq) agora atua como um "cérebro" (Agente ReAct)
que decide quando usar a "ferramenta" de busca (nosso pipeline V2).

ATUALIZAÇÕES V3:
- Implementa a lógica de "Tool Calling" do Groq.
- Cria o `SYSTEM_PROMPT_AGENT`, que instrui o LLM a usar a ferramenta.
- Cria `run_agent_loop` para gerenciar o loop de Raciocínio-Ação.
- O pipeline V2 (RAG-Fusion + RRF) agora é uma ferramenta (`tool_buscar_na_clt`).
- Mantém a lógica de fallback de modelos que você criou.
"""

import os
import pickle
import json
import re
import string
import faiss
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from groq import Groq, GroqError
from dotenv import load_dotenv
import streamlit as st
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuração de Caminhos ---
BASE_DIR = r'C:\Users\mlzengo\Documents\Garden Solutions\Adicao_Contabilidade\Sprint I'
PATH_FAISS_INDEX = os.path.join(BASE_DIR, 'data', 'indices', 'clt_faiss.index')
PATH_BM25_INDEX = os.path.join(BASE_DIR, 'data', 'indices', 'clt_bm25.pkl')
PATH_MAPPING_FILE = os.path.join(BASE_DIR, 'data', 'indices', 'clt_chunks_mapped.json')
PATH_ENV_FILE = os.path.join(BASE_DIR, 'secrets', '.env')
MODELO_EMBEDDING = 'paraphrase-multilingual-mpnet-base-v2'

# --- Modelos Groq ---
# Mantendo o modelo que você definiu na V2
LLM_MODEL_NAME = 'llama-3.1-70b-versatile'
# Mantendo sua lista de fallback
FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

# --- (NOVO) Definição do Cérebro do Agente (V3) ---
SYSTEM_PROMPT_AGENT = """
Você é um assistente jurídico especialista em Direito do Trabalho brasileiro.

SUA TAREFA:
1.  Analise a PERGUNTA DO USUÁRIO.
2.  **Decida (Raciocine):**
    a) Se a pergunta for simples (cumprimentos como "Olá", "Obrigado", ou perguntas sobre quem você é), responda diretamente com base no seu conhecimento.
    b) Se a pergunta for técnica sobre direito do trabalho, CLT, leis, férias, demissão, etc., você DEVE usar a ferramenta `buscar_na_clt` para encontrar o contexto jurídico.
3.  **Após usar a ferramenta:**
    a) Você receberá o contexto (Artigos da CLT).
    b) Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE nesse contexto.
    c) Sempre cite a fonte (ex: "CLT, Art. 482").
    d) Se o contexto retornado pela ferramenta for "Nenhum contexto encontrado." ou não for relevante, responda APENAS: "Não encontrei informações sobre este tópico específico na CLT ou nas normas fornecidas para fornecer uma resposta."
4.  Nunca use seu conhecimento prévio sobre leis. Sua única fonte de verdade para leis é a ferramenta `buscar_na_clt`.
"""

# ---  Definição da Ferramenta do Agente  ---
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
# Lista de ferramentas que o agente pode usar
AGENT_TOOLS = [TOOL_BUSCAR_CLT]

# ---------------------------------

@st.cache_resource
def load_all_resources():
    # ... (Esta função permanece idêntica à V2) ...
    logger.info("--- INICIANDO CARREGAMENTO DE RECURSOS (V3) ---")
    load_dotenv(PATH_ENV_FILE)
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords')
    stopwords_pt = set(nltk.corpus.stopwords.words('portuguese'))
    stopwords_pt.update(list(string.punctuation))
    with open(PATH_MAPPING_FILE, 'r', encoding='utf-8') as f:
        chunks_mapeados = json.load(f)
    embedding_model = SentenceTransformer(MODELO_EMBEDDING)
    faiss_index = faiss.read_index(PATH_FAISS_INDEX)
    with open(PATH_BM25_INDEX, 'rb') as f:
        bm25_index = pickle.load(f)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: st.error("Erro: GROQ_API_KEY não encontrada."); st.stop()
    groq_client = Groq(api_key=api_key)
    logger.info("--- TODOS OS RECURSOS FORAM CARREGADOS ---")
    return stopwords_pt, chunks_mapeados, embedding_model, faiss_index, bm25_index, groq_client

# --- Funções Internas de Busca  ---

def preprocessar_query_bm25(texto: str, stopwords: set) -> list:
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = texto.split()
    return [token for token in tokens if token not in stopwords]

def expandir_query_com_llm(query: str, groq_client: Groq) -> list:
    """(V2) Usa o LLM para gerar variações da pergunta."""
    logger.info(f"Expandindo query: '{query}'")
    system_prompt = "Você é um especialista em direito do trabalho brasileiro. Sua tarefa é reescrever a pergunta do usuário de 3 formas diferentes, usando sinônimos e termos jurídicos técnicos (como 'abono pecuniário', 'rescisão', 'adicional de insalubridade', etc.) para melhorar os resultados de busca em uma base de dados legal. Retorne apenas as 3 novas perguntas, separadas por ponto e vírgula (;). Sem cabeçalhos ou numeração."
    user_prompt = f"Pergunta original: {query}"
    try:
        # Usamos o modelo mais rápido para esta tarefa
        response = groq_client.chat.completions.create(
            model='llama3-8b-8192', # Usando um modelo rápido e atual
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.5, max_tokens=256
        )
        novas_queries_str = response.choices[0].message.content
        novas_queries = [q.strip() for q in novas_queries_str.split(';') if q.strip()]
        todas_as_queries = [query] + novas_queries
        logger.info(f"Queries expandidas: {todas_as_queries}")
        return todas_as_queries
    except Exception as e:
        logger.error(f"Erro ao expandir query: {e}. Usando apenas a query original.")
        return [query]

def get_faiss_results(query: str, k: int = 10) -> dict:
    _, _, embed_model, faiss_idx, _, _ = load_all_resources()
    query_vector = embed_model.encode([query]).astype('float32')
    _, indices_faiss = faiss_idx.search(query_vector, k)
    return {int(idx): i for i, idx in enumerate(indices_faiss[0])}

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
            score_rrf = 1.0 / (k_rrf + rank)
            fused_scores[idx] = fused_scores.get(idx, 0.0) + score_rrf
    return fused_scores

def _execute_hybrid_rrf_search(queries: list, k: int = 5) -> tuple[list, str]:
    """(V2) Executa a busca híbrida (RRF) para MÚLTIPLAS queries."""
    stopwords, chunks_map, _, _, _, _ = load_all_resources()
    log_busca = "Iniciando busca RAG-Fusion (V2)...\n"
    log_busca += f"Queries Geradas: {queries}\n\n"
    
    todos_ranks_faiss = [get_faiss_results(q) for q in queries]
    todos_ranks_bm25 = [get_bm25_results(q) for q in queries]

    scores_faiss_fused = fuse_rrf_results(todos_ranks_faiss)
    scores_bm25_fused = fuse_rrf_results(todos_ranks_bm25)
    
    final_faiss_ranks = {idx: i for i, (idx, score) in enumerate(sorted(scores_faiss_fused.items(), key=lambda item: item[1], reverse=True))}
    final_bm25_ranks = {idx: i for i, (idx, score) in enumerate(sorted(scores_bm25_fused.items(), key=lambda item: item[1], reverse=True))}
    
    todos_indices = set(final_faiss_ranks.keys()).union(set(final_bm25_ranks.keys()))
    fused_hybrid_scores = {}
    
    for idx in todos_indices:
        score = (1.0 / (60 + final_faiss_ranks[idx])) if idx in final_faiss_ranks else 0
        score += (1.0 / (60 + final_bm25_ranks[idx])) if idx in final_bm25_ranks else 0
        fused_hybrid_scores[idx] = score

    resultados_finais_idx = sorted(fused_hybrid_scores.keys(), key=lambda idx: fused_hybrid_scores[idx], reverse=True)
    chunks_finais = [chunks_map[idx] for idx in resultados_finais_idx[:k]]
    
    log_busca += "\n--- Contexto Final Selecionado (Pós-RAG-Fusion) ---\n"
    if not chunks_finais:
        log_busca += "Nenhum artigo relevante encontrado."
    else:
        for i, chunk in enumerate(chunks_finais):
             log_busca += f"  - Rank {i+1}: {chunk['artigo_numero']} (Score RRF: {fused_hybrid_scores[resultados_finais_idx[i]]:.6f})\n"
             
    return chunks_finais, log_busca

def formatar_contexto_para_llm(chunks: list) -> str:
    if not chunks: return "Nenhum contexto encontrado."
    contexto_str = "--- CONTEXTO JURÍDICO (CLT E NORMAS CORRELATAS) ---\n\n"
    for i, chunk in enumerate(chunks):
        contexto_str += f"[FONTE {i+1}]\nOrigem: {chunk.get('documento_fonte', 'CLT')}\n"
        if chunk.get('titulo_clt'): contexto_str += f"Título: {chunk.get('titulo_clt')}\n"
        if chunk.get('capitulo_secao'): contexto_str += f"Capítulo/Seção: {chunk.get('capitulo_secao')}\n"
        contexto_str += f"Artigo: {chunk.get('artigo_numero')}\nTexto:\n```\n{chunk.get('conteudo_chunk')}\n```\n\n"
    return contexto_str

# --- Função de Execução da Ferramenta  ---
def tool_buscar_na_clt(pergunta: str) -> tuple[str, str]:
    """
    Função wrapper que executa todo o pipeline V2 como uma única ferramenta.
    Retorna o contexto formatado e o log de debug.
    """
    logger.info(f"[Agente] Ferramenta 'tool_buscar_na_clt' ativada com a pergunta: '{pergunta}'")
    _, _, _, _, _, groq_client = load_all_resources()
    
    # 1. RAG-Fusion (Expansão)
    queries = expandir_query_com_llm(pergunta, groq_client)
    
    # 2. Busca Híbrida (RRF)
    chunks, log_busca = _execute_hybrid_rrf_search(queries)
    
    # 3. Formatar Contexto
    contexto_str = formatar_contexto_para_llm(chunks)
    
    logger.info("[Agente] Busca concluída. Retornando contexto formatado para o LLM.")
    return contexto_str, log_busca

# --- O Cérebro do Agente  ---
def run_agent_loop(query: str, chat_history: list) -> tuple[str, str]:
    """
    Gerencia o loop ReAct (Reason-Act) do agente.
    Substitui a antiga `gerar_resposta_advogado`.
    """
    _, _, _, _, _, groq_client = load_all_resources()
    
    # Lista de modelos para tentar (padrão primeiro, depois fallbacks)
    models_to_try = [LLM_MODEL_NAME] + [m for m in FALLBACK_MODELS if m != LLM_MODEL_NAME]
    
    log_de_busca = "" # Log para o expander do Streamlit
    
    # O histórico para esta chamada específica
    messages = [{"role": "system", "content": SYSTEM_PROMPT_AGENT}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": query})

    last_error = None
    response_message = None
    
    for model_name in models_to_try:
        try:
            logger.info(f"[Agente] Loop 1: Enviando pergunta ao LLM ({model_name}) para decisão...")
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto", # Deixa o LLM decidir se usa a ferramenta
                temperature=0.0
            )
            response_message = response.choices[0].message
            logger.info(f"[Agente] Decisão recebida do LLM ({model_name}).")
            # Se funcionou, sai do loop de modelos
            break 
            
        except GroqError as e:
            last_error = e
            logger.warning(f"Erro (Loop 1) ao usar modelo {model_name}: {e}")
            # (Lógica de fallback que você criou)
            if e.code in ['model_decommissioned', 'invalid_request_error', 'model_not_found']:
                logger.info(f"Modelo {model_name} está descontinuado. Tentando próximo...")
                continue
            elif e.code == 'rate_limit_exceeded' or e.status_code == 429:
                logger.warning("Rate limit ou quota excedida. Tentando próximo modelo...")
                continue
            else:
                logger.warning(f"Erro inesperado com modelo {model_name}. Tentando próximo modelo...")
                continue
        except Exception as e:
            last_error = e
            logger.error(f"Erro não-Groq inesperado (Loop 1): {e}. Tentando próximo modelo...")
            continue
    
    # Se todos os modelos falharam na primeira chamada
    if response_message is None:
        logger.error(f"Erro ao chamar a API do Groq (Loop 1) com todos os modelos: {last_error}")
        return f"❌ Erro ao processar a solicitação (Loop 1).\n\nÚltimo erro: {last_error}", ""

    # Adiciona a resposta (decisão) do LLM ao histórico
    messages.append(response_message)
    
    # --- Verificação de Ação (Tool Call) ---
    
    if response_message.tool_calls:
        # O LLM decidiu usar uma ferramenta
        logger.info("[Agente] O LLM decidiu usar uma ferramenta.")
        tool_call = response_message.tool_calls[0]
        tool_name = tool_call.function.name
        
        if tool_name == "tool_buscar_na_clt":
            # Extrai os argumentos
            tool_args = json.loads(tool_call.function.arguments)
            pergunta_para_busca = tool_args.get("pergunta")
            
            # Executa a ferramenta (nosso pipeline V2)
            contexto_str, log_de_busca = tool_buscar_na_clt(pergunta=pergunta_para_busca)
            
            # Adiciona o resultado da ferramenta ao histórico
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": contexto_str,
            })
            
            # --- Loop 2: Geração da Resposta Final ---
            logger.info("[Agente] Loop 2: Enviando contexto da ferramenta para geração final...")
            
            last_error_loop2 = None
            final_response = None
            
            for model_name in models_to_try:
                try:
                    # Chama o LLM novamente, agora com o contexto da ferramenta
                    final_response = groq_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=2048
                    )
                    logger.info(f"[Agente] Resposta final recebida do LLM ({model_name}).")
                    break # Sucesso
                
                except GroqError as e:
                    last_error_loop2 = e
                    logger.warning(f"Erro (Loop 2) ao usar modelo {model_name}: {e}")
                    if e.code in ['model_decommissioned', 'invalid_request_error', 'model_not_found']:
                        logger.info(f"Modelo {model_name} está descontinuado. Tentando próximo...")
                        continue
                    # ... (outras lógicas de fallback)
                except Exception as e:
                    last_error_loop2 = e
                    logger.error(f"Erro não-Groq inesperado (Loop 2): {e}. Tentando próximo modelo...")
                    continue
            
            if final_response:
                return final_response.choices[0].message.content, log_de_busca
            else:
                logger.error(f"Erro ao chamar a API do Groq (Loop 2) com todos os modelos: {last_error_loop2}")
                return f"❌ Erro ao processar a solicitação (Loop 2).\n\nÚltimo erro: {last_error_loop2}", log_de_busca
                
    else:
        # O LLM decidiu responder diretamente (sem ferramentas)
        logger.info("[Agente] O LLM decidiu responder diretamente.")
        return response_message.content, "(Nenhuma busca na CLT foi necessária)"

# --- Interface Gráfica ---
def main():
    st.set_page_config(
        page_title="Assistente Jurídico CLT",
        page_icon="⚖️",
        layout="centered"
    )
    
    st.title("⚖️ Agente Trabalhista CLT (V3 - Agente)")
    st.caption("Este agente decide ativamente quando usar a busca RAG-Fusion na CLT.")

    with st.spinner("Carregando cérebro do agente (V3)..."):
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
            
            # Prepara o histórico para o Agente (sem o prompt do sistema)
            history_for_agent = [
                msg for msg in st.session_state.messages 
                if msg["role"] in ("user", "assistant")
            ]
            history_for_agent = history_for_agent[:-1] 

            resposta, log_busca = run_agent_loop(prompt, history_for_agent)
            
            with st.expander("Ver Raciocínio do Agente (Debug V3)"):
                st.text(log_busca)

            message_placeholder.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})

if __name__ == "__main__":
    main()