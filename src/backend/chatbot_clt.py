"""
FASE 3 (Versão V3.1): Agente RAG Híbrido (Correção de Erros)
=========================================================

ATUALIZAÇÕES V3.1:
- Corrige o `AttributeError` no bloco `except GroqError`.
- Atualiza a lista de modelos para `llama-3.1-8b-instant`, que foi
  o único modelo confirmado como FUNCIONAL pelos logs do usuário.
- Remove os outros modelos descontinuados da lista de fallback.
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

# --- (CORREÇÃO) Modelos Groq ---
# Baseado nos logs de 17:06:00, este foi o único modelo que retornou 200 OK.
LLM_MODEL_NAME = 'llama-3.1-8b-instant' 
# Outros modelos que você pode tentar (mas que falharam nos logs anteriores)
FALLBACK_MODELS = ['llama3-8b-8192', 'gemma-7b-it']
# ---------------------------------

# --- Definição do Agente e Ferramentas (V3) ---
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
    logger.info("--- INICIANDO CARREGAMENTO DE RECURSOS (V3.1) ---")
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

# --- Funções Internas de Busca (V2) ---

def preprocessar_query_bm25(texto: str, stopwords: set) -> list:
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = texto.split()
    return [token for token in tokens if token not in stopwords]

def expandir_query_com_llm(query: str, groq_client: Groq) -> list:
    """
    (V3.2) Usa o LLM para gerar variações da pergunta, com foco explícito
    na tradução de termos coloquiais para jurídicos (few-shot).
    """
    logger.info(f"Expandindo query (V3.2): '{query}'")
    
    # Prompt de sistema (Prompt Engineering) aprimorado com exemplos (few-shot)
    system_prompt = """
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
    
    user_prompt = f"Pergunta original: {query}"
    
    try:
        # Usamos o modelo rápido e funcional
        response = groq_client.chat.completions.create(
            model=LLM_MODEL_NAME, # O 'llama-3.1-8b-instant' que funcionou
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=256
        )
        
        novas_queries_str = response.choices[0].message.content
        novas_queries = [q.strip() for q in novas_queries_str.split(';') if q.strip()]
        
        # Garante que as novas queries sejam únicas
        todas_as_queries = [query]
        for q in novas_queries:
            if q not in todas_as_queries:
                todas_as_queries.append(q)
        
        logger.info(f"Queries expandidas (V3.2): {todas_as_queries}")
        return todas_as_queries
        
    except Exception as e:
        logger.error(f"Erro ao expandir query (V3.2): {e}. Usando apenas a query original.")
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

# --- (CORREÇÃO) Função de Execução da Ferramenta (V3.1) ---
def tool_buscar_na_clt(pergunta: str) -> tuple[str, str]:
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

# --- (CORREÇÃO) O Cérebro do Agente (V3.1) ---
def run_agent_loop(query: str, chat_history: list) -> tuple[str, str]:
    _, _, _, _, _, groq_client = load_all_resources()
    
    # Lista de modelos para tentar (padrão é o único que funcionou no log)
    models_to_try = [LLM_MODEL_NAME] + [m for m in FALLBACK_MODELS if m != LLM_MODEL_NAME]
    
    log_de_busca = ""
    
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
                tool_choice="auto",
                temperature=0.0
            )
            response_message = response.choices[0].message
            logger.info(f"[Agente] Decisão recebida do LLM ({model_name}).")
            break 
            
        except GroqError as e:
            last_error = e
            # --- (CORREÇÃO) Lógica de verificação de erro ---
            # Converte o erro em string para verificar as mensagens
            error_msg = str(e).lower()
            logger.warning(f"Erro (Loop 1) ao usar modelo {model_name}: {error_msg}")
            
            model_errors = ["decommissioned", "no longer supported", "model_not_found"]
            
            # Verifica se alguma das strings de erro está na mensagem
            is_model_error = any(err in error_msg for err in model_errors)
            
            if is_model_error:
                logger.info(f"Modelo {model_name} está descontinuado. Tentando próximo...")
                continue
            elif "rate limit" in error_msg or "429" in error_msg or "quota" in error_msg:
                logger.warning("Rate limit ou quota excedida. Tentando próximo modelo...")
                continue
            else:
                logger.warning(f"Erro inesperado com modelo {model_name}. Tentando próximo modelo...")
                continue
        except Exception as e:
            last_error = e
            logger.error(f"Erro não-Groq inesperado (Loop 1): {e}. Tentando próximo modelo...")
            continue
    
    if response_message is None:
        logger.error(f"Erro ao chamar a API do Groq (Loop 1) com todos os modelos: {last_error}")
        return f"❌ Erro ao processar a solicitação (Loop 1).\n\nÚltimo erro: {last_error}", ""

    messages.append(response_message)
    
    # --- Verificação de Ação (Tool Call) ---
    
    if response_message.tool_calls:
        logger.info("[Agente] O LLM decidiu usar uma ferramenta.")
        tool_call = response_message.tool_calls[0]
        tool_name = tool_call.function.name
        
        if tool_name == "tool_buscar_na_clt":
            tool_args = json.loads(tool_call.function.arguments)
            pergunta_para_busca = tool_args.get("pergunta")
            
            contexto_str, log_de_busca = tool_buscar_na_clt(pergunta=pergunta_para_busca)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": contexto_str,
            })
            
            logger.info("[Agente] Loop 2: Enviando contexto da ferramenta para geração final...")
            
            last_error_loop2 = None
            final_response = None
            
            for model_name in models_to_try:
                try:
                    final_response = groq_client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=2048
                    )
                    logger.info(f"[Agente] Resposta final recebida do LLM ({model_name}).")
                    break
                
                except GroqError as e:
                    last_error_loop2 = e
                    # --- (CORREÇÃO) Lógica de verificação de erro ---
                    error_msg = str(e).lower()
                    logger.warning(f"Erro (Loop 2) ao usar modelo {model_name}: {error_msg}")
                    
                    model_errors = ["decommissioned", "no longer supported", "model_not_found"]
                    is_model_error = any(err in error_msg for err in model_errors)
                    
                    if is_model_error:
                        logger.info(f"Modelo {model_name} está descontinuado. Tentando próximo...")
                        continue
                    elif "rate limit" in error_msg or "429" in error_msg or "quota" in error_msg:
                        logger.warning("Rate limit ou quota excedida. Tentando próximo modelo...")
                        continue
                    else:
                        logger.warning(f"Erro inesperado com modelo {model_name}. Tentando próximo modelo...")
                        continue
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
        logger.info("[Agente] O LLM decidiu responder diretamente.")
        return response_message.content, "(Nenhuma busca na CLT foi necessária)"

# --- Interface Gráfica (Streamlit) ---
def main():
    st.set_page_config(
        page_title="Assistente Jurídico CLT",
        page_icon="⚖️",
        layout="centered"
    )
    
    st.title("⚖️ Agente Trabalhista CLT (V3.1 - Agente)")
    st.caption("Este agente decide ativamente quando usar a busca RAG-Fusion na CLT.")

    with st.spinner("Carregando cérebro do agente (V3.1)..."):
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
            
            with st.expander("Ver Raciocínio do Agente (Debug V3.1)"):
                st.text(log_busca)

            message_placeholder.markdown(resposta)
            st.session_state.messages.append({"role": "assistant", "content": resposta})

if __name__ == "__main__":
    main()