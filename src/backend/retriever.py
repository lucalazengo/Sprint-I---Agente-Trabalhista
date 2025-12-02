"""
Retriever Module
================
Handles the retrieval of relevant context from the knowledge base (CLT).
Implements Hybrid Search (BM25 + ChromaDB) with Reciprocal Rank Fusion (RRF),
Query Expansion, and Re-Ranking (Cross-Encoder).
"""

import pickle
import json
import string
import logging
import numpy as np
import chromadb
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from openai import OpenAI

# Import centralized config
from src.utils import config

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, llm_client: OpenAI = None):
        """
        Initialize the Retriever with necessary resources.
        """
        self.llm_client = llm_client
        self.stopwords = self._load_stopwords()
        self.chunks_map = self._load_chunks_map()
        self.bm25_index = self._load_bm25()
        self.chroma_collection = self._load_chroma()
        
        # Embeddings for Retrieval
        self.embed_model = SentenceTransformer(config.MODEL_EMBEDDING)
        
        # Re-Ranker Model (Lightweight Cross-Encoder)
        # Using ms-marco-MiniLM-L-6-v2 for speed/performance balance on CPU
        logger.info("Loading Re-Ranker Model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        logger.info("Retriever initialized successfully.")

    def _load_stopwords(self):
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        stopwords = set(nltk.corpus.stopwords.words('portuguese'))
        stopwords.update(list(string.punctuation))
        return stopwords

    def _load_chunks_map(self) -> Dict:
        if not config.PATH_MAPPING_FILE.exists():
            raise FileNotFoundError(f"Chunks map not found at {config.PATH_MAPPING_FILE}")
        with open(config.PATH_MAPPING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_bm25(self) -> BM25Okapi:
        if not config.PATH_BM25_INDEX.exists():
            raise FileNotFoundError(f"BM25 index not found at {config.PATH_BM25_INDEX}")
        with open(config.PATH_BM25_INDEX, 'rb') as f:
            return pickle.load(f)

    def _load_chroma(self) -> chromadb.Collection:
        client = chromadb.PersistentClient(
            path=str(config.PATH_CHROMA_DB),
            settings=chromadb.config.Settings(anonymized_telemetry=False)
        )
        return client.get_collection(name=config.CHROMA_COLLECTION_NAME)

    def preprocess_query_bm25(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()
        return [token for token in tokens if token not in self.stopwords]

    def expand_query(self, query: str) -> List[str]:
        """Expands the user query using the LLM to improve recall."""
        if not self.llm_client:
            logger.warning("LLM Client not provided. Skipping query expansion.")
            return [query]

        system_prompt = """
        Você é um especialista em direito do trabalho brasileiro. Sua tarefa é analisar a pergunta do usuário
        e gerar 3 variações de busca para encontrar os artigos de lei corretos.
        REGRAS:
        1. Traduza termos coloquiais para termos jurídicos.
        2. Inclua nomes de leis ou artigos principais se souber.
        3. Retorne APENAS as 3 variações separadas por ponto e vírgula (;).
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Pergunta original: {query}"}
                ],
                temperature=0.5,
                max_tokens=256
            )
            novas_queries_str = response.choices[0].message.content
            novas_queries = [q.strip() for q in novas_queries_str.split(';') if q.strip()]
            logger.info(f"Query Expansion: {novas_queries}")
            return [query] + novas_queries
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]

    def search_chroma(self, query: str, k: int = 10) -> Dict[str, int]:
        query_com_instrucao = f"Represente esta frase para buscar passagens relevantes: {query}"
        query_vector = self.embed_model.encode(query_com_instrucao, normalize_embeddings=True).tolist()
        results = self.chroma_collection.query(query_embeddings=[query_vector], n_results=k)
        
        if not results['ids']:
            return {}
            
        return {str(idx): i for i, idx in enumerate(results['ids'][0])}

    def search_bm25(self, query: str, k: int = 10) -> Dict[str, int]:
        query_tokens = self.preprocess_query_bm25(query)
        scores = self.bm25_index.get_scores(query_tokens)
        # Get top K indices
        top_indices = np.argsort(scores)[::-1][:k]
        return {str(idx): i for i, idx in enumerate(top_indices) if scores[idx] > 0}

    def fuse_results(self, rankings_list: List[Dict[str, int]], k_rrf: int = 60) -> Dict[str, float]:
        """Performs Reciprocal Rank Fusion (RRF) on multiple result sets."""
        fused_scores = {}
        for ranking in rankings_list:
            for doc_id, rank in ranking.items():
                score = 1.0 / (k_rrf + rank)
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score
        return fused_scores

    def hybrid_search(self, query: str, k: int = 5) -> Tuple[List[Dict], str]:
        """
        Executes the full hybrid search pipeline: 
        Expansion -> Search (Vector+Keyword) -> RRF Fusion -> RE-RANKING -> Final Top K
        """
        # 1. Expansion & Initial Retrieval (Broader Scope: 3x K)
        k_retrieval = k * 3 
        queries = self.expand_query(query)
        log_msg = f"Hybrid Search initiated with queries: {queries}\n"

        all_chroma_ranks = [self.search_chroma(q, k=k_retrieval) for q in queries]
        all_bm25_ranks = [self.search_bm25(q, k=k_retrieval) for q in queries]
        
        chroma_fused = self.fuse_results(all_chroma_ranks)
        bm25_fused = self.fuse_results(all_bm25_ranks)

        # RRF Combination
        all_ids = set(chroma_fused.keys()).union(set(bm25_fused.keys()))
        rrf_scores = {}
        
        for doc_id in all_ids:
            score = 0
            if doc_id in chroma_fused:
                score += chroma_fused[doc_id]
            if doc_id in bm25_fused:
                score += bm25_fused[doc_id]
            rrf_scores[doc_id] = score

        # Get Top Candidates for Re-Ranking (e.g., Top 15)
        sorted_candidates_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k_retrieval]
        
        candidates = []
        candidate_indices = []
        
        for doc_id in sorted_candidates_ids:
            try:
                chunk = self.chunks_map[int(doc_id)]
                candidates.append(chunk['conteudo_chunk'])
                candidate_indices.append(int(doc_id))
            except:
                continue
                
        if not candidates:
            return [], "No candidates found."

        # 2. Re-Ranking (Cross-Encoder)
        log_msg += f"\n--- Re-Ranking (Cross-Encoder) ---\n"
        log_msg += f"Re-ranking top {len(candidates)} candidates...\n"
        
        # Form pairs (Query, Document)
        pairs = [[query, doc] for doc in candidates]
        cross_scores = self.reranker.predict(pairs)
        
        # Sort by Cross-Encoder Score
        # Zip indices, scores, and original text
        scored_candidates = list(zip(candidate_indices, cross_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select Final Top K
        final_results = []
        for idx, score in scored_candidates[:k]:
            chunk = self.chunks_map[idx]
            final_results.append(chunk)
            log_msg += f"  - Rank {len(final_results)}: {chunk.get('artigo_numero')} (Re-Rank Score: {score:.4f})\n"

        return final_results, log_msg
