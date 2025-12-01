"""
Hybrid retrieval combining dense and sparse methods
"""
import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path

from rag.build_faiss import FAISSIndexBuilder
from utils.config import RAG_CONFIG, FAISS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid retrieval using dense (FAISS) and sparse (BM25) methods"""
    
    def __init__(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.3,
        gnn_weight: float = 0.2
    ):
        """
        Initialize hybrid retriever
        
        Args:
            dense_weight: Weight for dense retrieval
            sparse_weight: Weight for sparse retrieval
            gnn_weight: Weight for GNN-based retrieval
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.gnn_weight = gnn_weight
        
        # Initialize retrievers
        self.faiss_retriever = FAISSIndexBuilder()
        self.bm25 = None
        self.tokenized_corpus = None
    
    def load_dense_index(self, path: Optional[Path] = None):
        """Load FAISS index"""
        self.faiss_retriever.load_index(path)
        logger.info("Loaded dense index")
    
    def build_sparse_index(self, documents: Optional[List[Dict]] = None):
        """Build BM25 sparse index"""
        if documents is None:
            documents = self.faiss_retriever.documents
        
        # Tokenize documents
        self.tokenized_corpus = [
            doc['text'].lower().split()
            for doc in documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"Built BM25 index with {len(documents)} documents")
    
    def save_sparse_index(self, path: Optional[Path] = None):
        """Save BM25 index"""
        if path is None:
            path = FAISS_DIR / 'bm25_index.pkl'
        
        with open(path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'tokenized_corpus': self.tokenized_corpus
            }, f)
        
        logger.info(f"Saved BM25 index to {path}")
    
    def load_sparse_index(self, path: Optional[Path] = None):
        """Load BM25 index"""
        if path is None:
            path = FAISS_DIR / 'bm25_index.pkl'
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.tokenized_corpus = data['tokenized_corpus']
        
        logger.info(f"Loaded BM25 index from {path}")
    
    def dense_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Dense retrieval using FAISS"""
        results = self.faiss_retriever.search(query, top_k=top_k)
        return results
    
    def sparse_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Sparse retrieval using BM25"""
        if self.bm25 is None:
            logger.warning("BM25 index not built")
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Prepare results
        results = []
        documents = self.faiss_retriever.documents
        
        for rank, idx in enumerate(top_indices):
            if idx < len(documents):
                result = {
                    'document': documents[idx],
                    'score': float(scores[idx]),
                    'rank': rank + 1
                }
                results.append(result)
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_k: int = 10,
        sparse_k: int = 10
    ) -> List[Dict]:
        """
        Hybrid search combining dense and sparse retrieval
        
        Args:
            query: Query text
            top_k: Number of final results
            dense_k: Number of results from dense retrieval
            sparse_k: Number of results from sparse retrieval
        
        Returns:
            Ranked list of documents
        """
        # Get results from both methods
        dense_results = self.dense_search(query, top_k=dense_k)
        sparse_results = self.sparse_search(query, top_k=sparse_k)
        
        # Normalize scores
        dense_scores = self._normalize_scores([r['score'] for r in dense_results])
        sparse_scores = self._normalize_scores([r['score'] for r in sparse_results])
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for i, result in enumerate(dense_results):
            doc_id = id(result['document']['text'])
            combined_scores[doc_id] = {
                'document': result['document'],
                'score': self.dense_weight * dense_scores[i],
                'dense_rank': i + 1
            }
        
        # Add sparse scores
        for i, result in enumerate(sparse_results):
            doc_id = id(result['document']['text'])
            
            if doc_id in combined_scores:
                combined_scores[doc_id]['score'] += self.sparse_weight * sparse_scores[i]
                combined_scores[doc_id]['sparse_rank'] = i + 1
            else:
                combined_scores[doc_id] = {
                    'document': result['document'],
                    'score': self.sparse_weight * sparse_scores[i],
                    'sparse_rank': i + 1
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]
        
        # Add final rank
        for i, result in enumerate(sorted_results):
            result['rank'] = i + 1
        
        return sorted_results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1]"""
        if not scores:
            return []
        
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score < 1e-8:
            return [1.0] * len(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized.tolist()


def main():
    """Test hybrid retrieval"""
    from rag.build_corpus import CorpusBuilder
    from utils.config import BASE_DIR, CORPUS_DIR
    import json
    
    # Load corpus
    corpus_path = CORPUS_DIR / 'corpus.json'
    with open(corpus_path, 'r') as f:
        documents = json.load(f)
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        dense_weight=RAG_CONFIG['dense_weight'],
        sparse_weight=RAG_CONFIG['sparse_weight'],
        gnn_weight=RAG_CONFIG['gnn_weight']
    )
    
    # Load dense index
    retriever.load_dense_index()
    
    # Build sparse index
    retriever.build_sparse_index()
    retriever.save_sparse_index()
    
    # Test hybrid search
    query = "What are the sales trends for Store 1?"
    print(f"\nQuery: {query}\n")
    
    results = retriever.hybrid_search(query, top_k=3)
    
    for result in results:
        print(f"Rank {result['rank']} (Score: {result['score']:.3f}):")
        print(f"  Dense Rank: {result.get('dense_rank', 'N/A')}")
        print(f"  Sparse Rank: {result.get('sparse_rank', 'N/A')}")
        print(f"  Text: {result['document']['text'][:150]}...")
        print()


if __name__ == "__main__":
    main()
