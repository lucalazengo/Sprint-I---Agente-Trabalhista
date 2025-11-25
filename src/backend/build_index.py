"""
Index Builder
=============
Builds the Hybrid Search Indices (ChromaDB + BM25).
This script should be run when the underlying data (CLT chunks) changes.

Usage:
    python -m src.backend.build_index
"""

import json
import pickle
import re
import string
import logging
import nltk
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.utils import config

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IndexBuilder")

def load_chunks() -> list:
    """Load chunks from the processed JSON file."""
    if not config.PATH_MAPPING_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found at {config.PATH_MAPPING_FILE}")
        
    with open(config.PATH_MAPPING_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        logger.info(f"Loaded {len(data)} chunks.")
        return data

def prepare_stopwords():
    """Download and prepare Portuguese stopwords."""
    try: 
        nltk.data.find('corpora/stopwords')
    except LookupError: 
        nltk.download('stopwords')
        
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    stopwords_pt.extend(list(string.punctuation))
    # Add legal boilerplate terms that don't help with search
    stopwords_pt.extend(['art', 'artigo', 'lei', 'decreto', 'inciso', 'parágrafo', 'caput'])
    return set(stopwords_pt)

def preprocess_bm25(text: str, stopwords: set) -> list:
    """Tokenize and clean text for BM25."""
    text = text.lower()
    text = re.sub(r'(art|§|inc)\s*[\dº°ªivxlcma-z\-\.]+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b\d+\b', '', text)
    tokens = text.split()
    return [token for token in tokens if token not in stopwords and len(token) > 2]

def build_bm25(chunks: list):
    """Build and save BM25 index."""
    logger.info("Building BM25 index...")
    stopwords = prepare_stopwords()
    corpus = [preprocess_bm25(chunk['conteudo_chunk'], stopwords) for chunk in chunks]
    
    bm25 = BM25Okapi(corpus)
    
    config.PATH_BM25_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(config.PATH_BM25_INDEX, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 index saved to {config.PATH_BM25_INDEX}")

def build_chroma(chunks: list):
    """Build or update ChromaDB index."""
    logger.info(f"Building ChromaDB index with model {config.MODEL_EMBEDDING}...")
    
    model = SentenceTransformer(config.MODEL_EMBEDDING)
    texts = [chunk['conteudo_chunk'] for chunk in chunks]
    
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "origem": chunk.get('documento_fonte') or "N/A",
            "artigo": chunk.get('artigo_numero') or "N/A",
            "titulo_clt": chunk.get('titulo_clt') or "N/A",
            "capitulo_secao": chunk.get('capitulo_secao') or "N/A"
        })
        ids.append(str(i))

    client = chromadb.PersistentClient(path=str(config.PATH_CHROMA_DB))
    collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
    
    start_index = collection.count()
    if start_index >= len(texts):
        logger.info(f"Collection already has {start_index} items. Skipping.")
        return

    logger.info(f"Resuming indexing from item {start_index}...")
    batch_size = 16
    
    for i in range(start_index, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        logger.info(f"Processing batch {i // batch_size + 1}...")
        embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        
        collection.add(
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
            ids=batch_ids
        )
        
    logger.info(f"ChromaDB index saved to {config.PATH_CHROMA_DB}")

def main():
    chunks = load_chunks()
    build_bm25(chunks)
    build_chroma(chunks)
    logger.info("Index build complete.")

if __name__ == "__main__":
    main()
