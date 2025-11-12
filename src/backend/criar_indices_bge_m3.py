"""
FASE 2: Construtor de Índices BGE-M3 + ChromaDB
======================================================================
"""

import json
import os
import pickle
import re
import string
import numpy as np
import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuração de Caminhos ---
BASE_DIR = r'/Users/universo369/Documents/UNIVERSO_369/U369_root/GARDEN SOLUTION/adicao_contabilidade/Sprint-I---Agente-Trabalhista'
PATH_INPUT_CHUNKS = os.path.join(BASE_DIR, 'data', 'processed_data', 'clt_chunks.json') 
MODELO_EMBEDDING = 'BAAI/bge-m3'
PATH_OUTPUT_BM25 = os.path.join(BASE_DIR, 'data', 'processed_data', 'indices', 'clt_bm25_bge-m3.pkl')
PATH_OUTPUT_CHROMA_DB = os.path.join(BASE_DIR, 'data', 'processed_data', 'indices', 'chroma_db_bge_m3')
CHROMA_COLLECTION_NAME = "clt_bge_m3"
# ---------------------------------

def carregar_chunks(filepath: str) -> list:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Sucesso: {len(data)} chunks carregados de {filepath}")
            return data
    except Exception as e:
        logger.error(f"Erro ao carregar chunks: {e}")
        raise

def preparar_stopwords_portuguesas():
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords')
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    stopwords_pt.extend(list(string.punctuation))
    stopwords_pt.extend(['art', 'artigo', 'lei', 'decreto', 'inciso', 'parágrafo', 'caput'])
    return set(stopwords_pt)

def preprocessar_para_bm25(texto: str, stopwords: set) -> list:
    texto = texto.lower()
    texto = re.sub(r'(art|§|inc)\s*[\dº°ªivxlcma-z\-\.]+', ' ', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = re.sub(r'\b\d+\b', '', texto)
    tokens = texto.split()
    return [token for token in tokens if token not in stopwords and len(token) > 2]

def criar_indice_bm25(chunks: list, stopwords: set, output_path: str):
    logger.info("Iniciando criação do índice BM25...")
    corpus_bm25 = [preprocessar_para_bm25(chunk['conteudo_chunk'], stopwords) for chunk in chunks]
    bm25 = BM25Okapi(corpus_bm25)
    with open(output_path, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"Sucesso: Índice BM25 (bge-m3) salvo em {output_path}")

# --- (FUNÇÃO ATUALIZADA V2.4 - COM LÓGICA DE RESUMO) ---
def criar_ou_atualizar_indice_chroma(chunks: list, model_name: str, output_path: str, collection_name: str):
    logger.info("Iniciando criação/atualização do índice ChromaDB com BGE-M3 (em lotes)...")
    try:
        logger.info(f"Carregando modelo de embedding: {model_name}...")
        model = SentenceTransformer(model_name)
        logger.info("Modelo BGE-M3 carregado.")

        textos = [chunk['conteudo_chunk'] for chunk in chunks]
        
        # Prepara os metadados e IDs (V2.3 - correção de NoneType)
        metadatas_list = []
        ids_list = []
        for i, chunk in enumerate(chunks):
            meta = {
                "origem": chunk.get('documento_fonte') or "N/A",
                "artigo": chunk.get('artigo_numero') or "N/A",
                "titulo_clt": chunk.get('titulo_clt') or "N/A",
                "capitulo_secao": chunk.get('capitulo_secao') or "N/A"
            }
            metadatas_list.append(meta)
            ids_list.append(str(i)) # IDs devem ser strings

        # Inicializa o cliente ChromaDB
        client = chromadb.PersistentClient(path=output_path)
        
        #  Pega a coleção ou a cria. NÃO DELETA MAIS.
        collection = client.get_or_create_collection(name=collection_name)
        
        # --- (MUDANÇA V2.4) LÓGICA DE RESUMO ---
        # 1. Pega o número de itens JÁ indexados
        start_index = collection.count()
        
        if start_index >= len(textos):
            logger.info(f"A coleção '{collection_name}' já está completa com {start_index} itens. Nenhuma ação necessária.")
            return
        
        logger.info(f"A coleção contém {start_index} itens. Retomando a indexação do item {start_index}...")
        # --- FIM DA LÓGICA DE RESUMO ---

        batch_size = 16
        logger.info(f"Gerando embeddings (BGE-M3) em lotes de {batch_size}...")

        # --- (MUDANÇA V2.4) Loop agora começa de 'start_index' ---
        for i in range(start_index, len(textos), batch_size):
            # Define o lote atual
            text_batch = textos[i:i+batch_size]
            metadatas_batch = metadatas_list[i:i+batch_size]
            ids_batch = ids_list[i:i+batch_size]
            
            logger.info(f"Processando lote {i // batch_size + 1} / {len(textos) // batch_size + 1}...")
            
            # Gera embeddings APENAS para o lote
            embeddings_batch = model.encode(text_batch, show_progress_bar=False, normalize_embeddings=True)
            embeddings_list_batch = embeddings_batch.tolist()
            
            # Adiciona o lote ao ChromaDB
            collection.add(
                embeddings=embeddings_list_batch,
                documents=text_batch,
                metadatas=metadatas_batch,
                ids=ids_batch
            )
            logger.info(f"Lote {i // batch_size + 1} adicionado ao ChromaDB.")
        
        count = collection.count()
        logger.info(f"Sucesso: {count} vetores (BGE-M3) adicionados ao índice.")
        logger.info(f"Índice ChromaDB (bge-m3) salvo em {output_path}")
        
    except Exception as e:
        logger.error(f"ERRO ao criar o índice ChromaDB (BGE-M3): {e}")
        raise

def main():
    logger.info("--- INICIANDO FASE 2 (REBUILD CHROMA V2.4 + BM25) ---")
    
    global BASE_DIR, PATH_INPUT_CHUNKS, PATH_OUTPUT_BM25, PATH_OUTPUT_CHROMA_DB
    
    # Validação do BASE_DIR
    if not os.path.exists(BASE_DIR):
        logger.warning(f"O BASE_DIR '{BASE_DIR}' parece não existir. Verifique o caminho.")
        alt_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if os.path.exists(alt_base_dir):
            logger.info(f"Usando caminho relativo: {alt_base_dir}")
            BASE_DIR = alt_base_dir
        else:
            logger.error("Caminho BASE_DIR não encontrado. Abortando.")
            return

    # Atualiza os caminhos globais com base no BASE_DIR
    PATH_INPUT_CHUNKS = os.path.join(BASE_DIR, 'data', 'processed_data', 'clt_chunks.json') 
    PATH_OUTPUT_BM25 = os.path.join(BASE_DIR, 'data', 'processed_data', 'indices', 'clt_bm25_bge-m3.pkl')
    PATH_OUTPUT_CHROMA_DB = os.path.join(BASE_DIR, 'data', 'processed_data', 'indices', 'chroma_db_bge_m3')

    output_dir = os.path.dirname(PATH_OUTPUT_CHROMA_DB)
    if not os.path.exists(output_dir):
        logger.info(f"Criando diretório de índices: {output_dir}")
        os.makedirs(output_dir)
    
    if not os.path.exists(PATH_INPUT_CHUNKS):
        logger.error(f"Arquivo de CHUNKS não encontrado em: {PATH_INPUT_CHUNKS}. Verifique o caminho.")
        return
        
    chunks = carregar_chunks(PATH_INPUT_CHUNKS)
    if not chunks:
        logger.error("Nenhum chunk foi carregado. Abortando.")
        return

    stopwords = preparar_stopwords_portuguesas()
    
    # O BM25 é rápido, então recriamos ele
    criar_indice_bm25(chunks, stopwords, PATH_OUTPUT_BM25)
    
    # O ChromaDB será atualizado ou retomado
    criar_ou_atualizar_indice_chroma(chunks, MODELO_EMBEDDING, PATH_OUTPUT_CHROMA_DB, CHROMA_COLLECTION_NAME)

    logger.info("--- REBUILD DA FASE 2 CONCLUÍDO (CHROMA + BM25) ---")
    logger.info("Novos artefatos gerados. Você já pode rodar o app V4.1 com OpenAI e ChromaDB.")

if __name__ == "__main__":
    main()