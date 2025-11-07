"""
FASE 3: SERVIDOR RAG HÍBRIDO (FLASK + GROQ)
===================================================

Este script implementa a arquitetura V1 (RAG Híbrido + Hierárquico)
usando a API de inferência rápida do Groq (com o modelo Mixtral).
"""
import os
import pickle
import json
import re
import string
import numpy as np
import faiss
import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from groq import Groq  # Importa a biblioteca Groq
from dotenv import load_dotenv # Importa o dotenv
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuração de Caminhos ---
# --- Configuração de Caminhos ---
# Define o caminho base do seu projeto para referência
BASE_DIR = r'C:\Users\mlzengo\Documents\Garden Solutions\Adicao_Contabilidade\Sprint I'

# Caminhos absolutos para os arquivos de índice e .env
# --- Configuração de Caminhos ---
# Define o caminho base do seu projeto para referência
BASE_DIR = r'C:\Users\mlzengo\Documents\Garden Solutions\Adicao_Contabilidade\Sprint I'

# Caminhos absolutos para os arquivos de índice e .env
PATH_FAISS_INDEX = os.path.join(BASE_DIR, 'data', 'indices', 'clt_faiss.index')
PATH_BM25_INDEX = os.path.join(BASE_DIR, 'data', 'indices', 'clt_bm25.pkl')
PATH_MAPPING_FILE = os.path.join(BASE_DIR, 'data', 'indices', 'clt_chunks_mapped.json')
PATH_ENV_FILE = os.path.join(BASE_DIR, 'secrets', '.env')

# Modelo de embedding escolhido (multilíngue, ótimo para português)
MODELO_EMBEDDING = 'paraphrase-multilingual-mpnet-base-v2'
# ---------------------------------

# Caminho para seu arquivo .env, conforme você especificou
PATH_ENV_FILE = r'C:\Users\mlzengo\Documents\Garden Solutions\Adicao_Contabilidade\Sprint I\secrets\.env'
# ---------------------------------

# --- Carregamento Global dos Modelos e Índices ---
try:
    # 0. Carregar a API Key (NOVO)
    logger.info(f"Carregando variáveis de ambiente de {PATH_ENV_FILE}...")
    if not os.path.exists(PATH_ENV_FILE):
        raise FileNotFoundError(f"Arquivo .env não encontrado no caminho: {PATH_ENV_FILE}")
    load_dotenv(PATH_ENV_FILE)
    
    # 1. Carregar stopwords
    nltk.data.find('corpora/stopwords')
    stopwords_pt = set(nltk.corpus.stopwords.words('portuguese'))
    stopwords_pt.update(list(string.punctuation))
    
    # 2. Carregar o Mapeamento de Chunks
    logger.info(f"Carregando mapeamento de chunks de {PATH_MAPPING_FILE}...")
    with open(PATH_MAPPING_FILE, 'r', encoding='utf-8') as f:
        CHUNKS_MAPEADOS = json.load(f)
    logger.info(f"{len(CHUNKS_MAPEADOS)} chunks carregados.")

    # 3. Carregar o Modelo de Embedding
    logger.info(f"Carregando modelo de embedding: {MODELO_EMBEDDING}...")
    EMBEDDING_MODEL = SentenceTransformer(MODELO_EMBEDDING)
    logger.info("Modelo de embedding carregado.")

    # 4. Carregar Índice FAISS
    logger.info(f"Carregando índice FAISS de {PATH_FAISS_INDEX}...")
    FAISS_INDEX = faiss.read_index(PATH_FAISS_INDEX)
    logger.info("Índice FAISS carregado.")

    # 5. Carregar Índice BM25
    logger.info(f"Carregando índice BM25 de {PATH_BM25_INDEX}...")
    with open(PATH_BM25_INDEX, 'rb') as f:
        BM25_INDEX = pickle.load(f)
    logger.info("Índice BM25 carregado.")
    
    # 6. Configurar o LLM (Groq) (MODIFICADO)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Variável de ambiente GROQ_API_KEY não encontrada. Verifique seu arquivo .env.")
    
    GROQ_CLIENT = Groq(api_key=api_key)
    
    # Modelo padrão do Groq (modelos atualizados em 2025)
    # Usa llama-3.1-70b-versatile como padrão (mais robusto para tarefas jurídicas)
    # Se falhar, o sistema tentará automaticamente outros modelos (fallback)
    LLM_MODEL_NAME = "llama-3.1-70b-versatile"
    logger.info(f"LLM (Groq) configurado com o modelo padrão: {LLM_MODEL_NAME}")
    logger.info("Nota: Se este modelo falhar, o sistema tentará automaticamente modelos alternativos.")

    # 7. Inicializar o Flask
    app = Flask(__name__)
    logger.info("Servidor Flask inicializado.")

except Exception as e:
    logger.error(f"ERRO CRÍTICO DURANTE A INICIALIZAÇÃO: {e}")
    raise

# --- Funções de Pré-processamento e Busca (Idênticas ao script anterior) ---

def preprocessar_query_bm25(texto: str) -> list:
    """Limpa e tokeniza a *pergunta do usuário* para a busca BM25."""
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    tokens = texto.split()
    tokens_limpos = [token for token in tokens if token not in stopwords_pt]
    return tokens_limpos

def busca_hibrida(query: str, k_faiss: int = 3, k_bm25: int = 3) -> list:
    """Executa a busca híbrida (V1) e retorna os chunks de contexto."""
    
    resultados_combinados = {}
    
    # --- 1. Busca Semântica (FAISS) ---
    logger.info(f"[Busca Híbrida] Executando busca FAISS para: '{query}'")
    query_vector = EMBEDDING_MODEL.encode([query]).astype('float32')
    distancias, indices_faiss = FAISS_INDEX.search(query_vector, k_faiss)
    
    for i, idx in enumerate(indices_faiss[0]):
        idx = int(idx)
        if idx < len(CHUNKS_MAPEADOS):
            chunk = CHUNKS_MAPEADOS[idx]
            score = 1.0 / (1.0 + distancias[0][i]) 
            resultados_combinados[idx] = (chunk, score)
            logger.info(f"[FAISS] Encontrado: {chunk['artigo_numero']} (Score: {score:.4f})")
            
    # --- 2. Busca por Keyword (BM25) ---
    logger.info(f"[Busca Híbrida] Executando busca BM25 para: '{query}'")
    query_bm25 = preprocessar_query_bm25(query)
    scores_bm25 = BM25_INDEX.get_scores(query_bm25)
    
    indices_bm25 = np.argsort(scores_bm25)[::-1][:k_bm25] 
    
    for idx in indices_bm25:
        idx = int(idx)
        if idx < len(CHUNKS_MAPEADOS):
            chunk = CHUNKS_MAPEADOS[idx]
            score = scores_bm25[idx]
            if score <= 0: continue
            logger.info(f"[BM25] Encontrado: {chunk['artigo_numero']} (Score: {score:.4f})")
            if idx in resultados_combinados:
                chunk_existente, score_existente = resultados_combinados[idx]
                resultados_combinados[idx] = (chunk_existente, score_existente + score)
            else:
                resultados_combinados[idx] = (chunk, score)

    # --- 3. Re-ranking (Fusão) ---
    resultados_finais = sorted(resultados_combinados.values(), key=lambda x: x[1], reverse=True)
    chunks_finais = [chunk for chunk, score in resultados_finais]
    return chunks_finais[:k_faiss]

def formatar_contexto_para_llm(chunks: list) -> str:
    """Formata os artigos encontrados em uma string de contexto para o prompt."""
    if not chunks:
        return "Nenhum contexto encontrado."
    
    contexto_str = "--- CONTEXTO JURÍDICO (CLT E NORMAS CORRELATAS) ---\n\n"
    
    for i, chunk in enumerate(chunks):
        contexto_str += f"[FONTE {i+1}]\n"
        contexto_str += f"Origem: {chunk.get('documento_fonte', 'CLT')}\n"
        if chunk.get('titulo_clt'):
            contexto_str += f"Título: {chunk.get('titulo_clt')}\n"
        if chunk.get('capitulo_secao'):
            contexto_str += f"Capítulo/Seção: {chunk.get('capitulo_secao')}\n"
        contexto_str += f"Artigo: {chunk.get('artigo_numero')}\n"
        contexto_str += "Texto:\n"
        contexto_str += f"```\n{chunk.get('conteudo_chunk')}\n```\n\n"
    
    return contexto_str

# --- Função de Geração (MODIFICADA PARA GROQ) ---

def gerar_resposta_advogado(query: str, contexto: str) -> str:
    """Monta o Prompt Mestre e chama o LLM (Groq)."""
    
    # O "Prompt Mestre" agora é a mensagem de "system"
    system_prompt = f"""
    Você é um assistente jurídico especialista em Direito do Trabalho brasileiro.
    Sua única fonte de conhecimento é o CONTEXTO JURÍDICO fornecido abaixo.
    Você NUNCA deve usar conhecimento prévio ou externo.
    
    SUA TAREFA:
    1.  Analise a PERGUNTA DO USUÁRIO.
    2.  Analise o CONTEXTO JURÍDICO.
    3.  Responda à pergunta de forma objetiva, técnica e clara, baseando-se EXCLUSIVAMENTE nos artigos fornecidos no contexto.
    4.  Sempre cite a fonte exata da sua resposta, referenciando o Artigo e a Fonte (ex: "CLT, Art. 482").
    5.  Se o contexto fornecido não for suficiente para responder à pergunta, responda APENAS: "Não encontrei informações sobre este tópico específico na CLT ou nas normas fornecidas para fornecer uma resposta."
    6.  Não faça suposições, não dê opiniões pessoais e não ofereça conselhos jurídicos. Apenas informe o que a lei diz.
    """
    
    # O contexto e a pergunta formam a mensagem do "user"
    user_prompt = f"""
    {contexto}
    
    PERGUNTA DO USUÁRIO:
    "{query}"
    
    RESPOSTA DO ASSISTENTE JURÍDICO:
    """
    
    logger.info("Enviando prompt mestre para a API Groq...")
    
    # Lista de modelos de fallback caso o principal falhe
    # Modelos atualizados do Groq (2025) - ordenados por preferência
    fallback_models = [
        "llama-3.1-8b-instant",      # Modelo rápido, boa alternativa
        "llama-3.1-70b-versatile",   # Modelo robusto (já é o padrão, mas incluído como fallback)
        "gemma2-9b-it",              # Modelo alternativo do Google
        "mixtral-8x7b-32768",        # Modelo alternativo (pode estar descontinuado)
    ]
    
    # Remove o modelo atual da lista de fallback para evitar tentar o mesmo modelo duas vezes
    fallback_models = [m for m in fallback_models if m != LLM_MODEL_NAME]
    
    # Lista completa de modelos para tentar (modelo padrão primeiro, depois fallbacks)
    models_to_try = [LLM_MODEL_NAME] + fallback_models
    
    last_error = None
    for model_name in models_to_try:
        try:
            # Chama a API do Groq usando o formato de chat
            response = GROQ_CLIENT.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # Temperatura 0.0 para respostas factuais e objetivas
                max_tokens=2048  # Aumentado para respostas mais completas
            )
            logger.info(f"Resposta recebida do Groq usando modelo: {model_name}")
            return response.choices[0].message.content
            
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            error_repr = repr(e).lower()
            logger.warning(f"Erro ao usar modelo {model_name}: {e}")
            
            # Verifica se o erro é sobre modelo descontinuado ou não suportado
            model_errors = [
                "decommissioned",
                "no longer supported",
                "has been decommissioned",
                "not supported",
                "invalid_request_error",
                "model_decommissioned"
            ]
            
            is_model_error = any(err in error_msg or err in error_repr for err in model_errors)
            
            if is_model_error:
                logger.info(f"Modelo {model_name} foi descontinuado ou não está disponível. Tentando próximo modelo...")
                continue
            # Se é rate limit, também tenta próximo modelo
            elif "rate limit" in error_msg or "429" in error_msg or "quota" in error_msg:
                logger.warning("Rate limit ou quota excedida. Tentando próximo modelo...")
                continue
            # Para outros erros, também tenta o próximo modelo (pode ser problema temporário)
            else:
                logger.warning(f"Erro inesperado com modelo {model_name}. Tentando próximo modelo...")
                continue
    
    # Se todos os modelos falharam
    logger.error(f"Erro ao chamar a API do Groq com todos os modelos tentados: {last_error}")
    return f"Erro ao processar a solicitação ao LLM. Todos os modelos falharam. Último erro: {last_error}"

# --- Endpoint da API Flask (Idêntico ao script anterior) ---

@app.route('/perguntar_clt', methods=['POST'])
def perguntar_clt():
    """
    Endpoint principal da API para responder perguntas sobre a CLT.
    Recebe um JSON: {"pergunta": "..."}
    Retorna um JSON: {"resposta": "..."}
    """
    try:
        data = request.json
        if not data or 'pergunta' not in data:
            return jsonify({"erro": "A requisição deve ser um JSON com a chave 'pergunta'."}), 400
            
        query = data['pergunta']
        logger.info(f"Recebida nova requisição /perguntar_clt: '{query}'")
        
        # 1. Busca Híbrida (V1)
        chunks_relevantes = busca_hibrida(query)
        
        # 2. Formatar Contexto
        contexto_formatado = formatar_contexto_para_llm(chunks_relevantes)
        
        # 3. Gerar Resposta (agora com Groq)
        resposta = gerar_resposta_advogado(query, contexto_formatado)
        
        # 4. Retornar
        return jsonify({"resposta": resposta})

    except Exception as e:
        logger.error(f"Erro inesperado no endpoint /perguntar_clt: {e}")
        return jsonify({"erro": f"Erro interno no servidor: {e}"}), 500

# --- Execução do Servidor ---

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)