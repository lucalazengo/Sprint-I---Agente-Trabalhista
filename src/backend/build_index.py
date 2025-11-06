"""
FASE 2: INDEXAÇÃO HÍBRIDA (FAISS + BM25)
=========================================================

Este script carrega os chunks estruturados e cria
dois índices de busca:

1.  FAISS (Vetorial/Semântico): Para entender a *intenção* da pergunta.
2.  BM25 (Keyword/Lexical): Para encontrar *termos técnicos* e artigos exatos.

Este é o núcleo da nossa arquitetura de RAG Híbrido + Hierárquico.

Autor: Garden Solutions 
"""

import json
import os
import pickle
import re
import string
import numpy as np
import faiss
import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuração de Caminhos ---
# Caminhos relativos ao diretório raiz do projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_data')

# Cria diretório de índices se não existir
INDICES_DIR = os.path.join(DATA_DIR, 'indices')
os.makedirs(INDICES_DIR, exist_ok=True)

PATH_INPUT_CHUNKS = os.path.join(PROCESSED_DIR, 'clt_chunks.json')
PATH_OUTPUT_FAISS = os.path.join(INDICES_DIR, 'clt_faiss.index')
PATH_OUTPUT_BM25 = os.path.join(INDICES_DIR, 'clt_bm25.pkl')
PATH_OUTPUT_MAPPING = os.path.join(INDICES_DIR, 'clt_chunks_mapped.json')
# ---------------------------------

# Modelo de embedding escolhido (multilíngue, ótimo para português)
MODELO_EMBEDDING = 'paraphrase-multilingual-mpnet-base-v2'

def carregar_chunks(filepath: str) -> list:
    """Carrega os chunks do arquivo JSON gerado na Fase 1."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"Sucesso: {len(chunks)} chunks carregados de {filepath}")
        return chunks
    except FileNotFoundError:
        logger.error(f"ERRO: Arquivo de chunks não encontrado: {filepath}")
        raise
    except json.JSONDecodeError:
        logger.error(f"ERRO: O arquivo {filepath} não é um JSON válido.")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar chunks: {e}")
        raise

def preparar_stopwords_portuguesas():
    """Baixa e prepara a lista de stopwords em português."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Baixando lista de stopwords do NLTK...")
        nltk.download('stopwords')
    
    # Adiciona pontuação à lista de stopwords
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    stopwords_pt.extend(list(string.punctuation))
    # Adiciona termos comuns de "juridiquês" que não agregam valor à busca por keyword
    stopwords_pt.extend(['art', 'artigo', 'lei', 'decreto', 'inciso', 'parágrafo', 'caput'])
    
    return set(stopwords_pt)

def preprocessar_para_bm25(texto: str, stopwords: set) -> list:
    """Limpa e tokeniza o texto para o índice BM25."""
    texto = texto.lower()
    # Remove números de artigos, parágrafos, etc. para focar no conteúdo
    texto = re.sub(r'(art|§|inc)\s*[\dº°ªivxlcma-z\-\.]+', ' ', texto)
    # Remove pontuação
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    # Remove números soltos
    texto = re.sub(r'\b\d+\b', '', texto)
    # Tokeniza
    tokens = texto.split()
    # Remove stopwords
    tokens_limpos = [token for token in tokens if token not in stopwords and len(token) > 2]
    return tokens_limpos

def criar_indice_bm25(chunks: list, stopwords: set, output_path: str):
    """Cria e salva o índice BM25 (keyword)."""
    logger.info("Iniciando criação do índice BM25...")
    
    # 1. Prepara o "corpus" (lista de textos tokenizados)
    corpus_bm25 = [preprocessar_para_bm25(chunk['conteudo_chunk'], stopwords) for chunk in chunks]
    
    # 2. Treina o modelo BM25
    bm25 = BM25Okapi(corpus_bm25)
    
    # 3. Salva o modelo treinado em disco
    with open(output_path, 'wb') as f:
        pickle.dump(bm25, f)
        
    logger.info(f"Sucesso: Índice BM25 criado e salvo em {output_path}")

def criar_indice_faiss(chunks: list, model_name: str, output_path: str):
    """Cria e salva o índice FAISS (vetorial)."""
    logger.info("Iniciando criação do índice FAISS...")
    
    try:
        # 1. Carrega o modelo de embedding
        logger.info(f"Carregando modelo de embedding: {model_name}...")
        model = SentenceTransformer(model_name)
        logger.info("Modelo carregado.")

        # 2. Extrai os textos
        # Usamos o "conteudo_chunk" pois ele é o texto completo do artigo
        textos = [chunk['conteudo_chunk'] for chunk in chunks]

        # 3. Gera os embeddings (vetores)
        logger.info("Gerando embeddings... Isso pode levar alguns minutos.")
        embeddings = model.encode(textos, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32') # FAISS usa float32
        
        dimensao = embeddings.shape[1]
        logger.info(f"Embeddings gerados. Dimensão: {dimensao}")

        # 4. Cria o índice FAISS
        # IndexFlatL2 é um índice simples e rápido para busca de similaridade (distância L2)
        index = faiss.IndexFlatL2(dimensao)
        
        # 5. Adiciona os vetores ao índice
        index.add(embeddings)
        
        # 6. Salva o índice em disco
        faiss.write_index(index, output_path)
        
        logger.info(f"Sucesso: {index.ntotal} vetores adicionados ao índice FAISS.")
        logger.info(f"Índice FAISS salvo em {output_path}")
        
    except Exception as e:
        logger.error(f"ERRO ao criar o índice FAISS: {e}")
        raise

def salvar_mapeamento(chunks: list, output_path: str):
    """Salva os chunks, que agora servem como nosso mapeamento ID -> Conteúdo."""
    # O próprio arquivo de chunks já é o mapeamento.
    # O índice no FAISS/BM25 (0, 1, 2...) corresponde ao índice na lista de chunks.
    # Apenas copiamos/reescrevemos para ter um nome claro.
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Sucesso: Mapeamento de chunks salvo em {output_path}")
    except Exception as e:
        logger.error(f"ERRO ao salvar o arquivo de mapeamento: {e}")
        raise

def main():
    """Função principal para orquestrar a indexação híbrida."""
    logger.info("--- INICIANDO FASE 2: INDEXAÇÃO HÍBRIDA (V1) ---")
    
    # 1. Carregar chunks da Fase 1
    chunks = carregar_chunks(PATH_INPUT_CHUNKS)
    if not chunks:
        return

    # 2. Preparar stopwords para o BM25
    stopwords = preparar_stopwords_portuguesas()
    
    # 3. Criar e salvar índice BM25 (Keyword)
    criar_indice_bm25(chunks, stopwords, PATH_OUTPUT_BM25)
    
    # 4. Criar e salvar índice FAISS (Vetorial)
    criar_indice_faiss(chunks, MODELO_EMBEDDING, PATH_OUTPUT_FAISS)
    
    # 5. Salvar o arquivo de mapeamento (a lista de chunks)
    # Este arquivo é essencial para a Fase 3
    salvar_mapeamento(chunks, PATH_OUTPUT_MAPPING)

    logger.info("--- FASE 2 CONCLUÍDA ---")
    logger.info("Arquivos gerados:")
    logger.info(f"  - Índice Vetorial: {PATH_OUTPUT_FAISS}")
    logger.info(f"  - Índice Keyword:  {PATH_OUTPUT_BM25}")
    logger.info(f"  - Mapeamento:    {PATH_OUTPUT_MAPPING}")
    logger.info("Próximo passo: Fase 3 (Integração RAG e API).")

if __name__ == "__main__":
    main()