"""
Build FAISS index for vector retrieval
"""
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import json

from utils.config import FAISS_DIR, CORPUS_DIR, RAG_CONFIG, BASE_DIR
from utils.embedding import EmbeddingGenerator
from utils.logger import setup_logger

logger = setup_logger(__name__, BASE_DIR / "logs" / "faiss.log")


class FAISSIndexBuilder:
    """Build and manage FAISS index"""
    
    def __init__(self, embedding_dim: Optional[int] = None):
        """
        Initialize FAISS index builder
        
        Args:
            embedding_dim: Embedding dimension
        """
        self.embedding_dim = embedding_dim or RAG_CONFIG['embedding_dim']
        self.index = None
        self.documents = []
        self.embeddings = None
        self.embedding_generator = EmbeddingGenerator()
    
    def build_index(
        self,
        documents: List[Dict],
        index_type: str = 'flat'
    ):
        """
        Build FAISS index from documents
        
        Args:
            documents: List of documents with 'text' and 'metadata'
            index_type: 'flat' or 'ivf' (inverted file index)
        """
        logger.info(f"Building FAISS index for {len(documents)} documents...")
        
        self.documents = documents
        
        # Extract texts
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        self.embeddings = self.embedding_generator.encode_texts(
            texts,
            method='sentence_transformer'
        )
        
        # Update embedding dimension
        self.embedding_dim = self.embeddings.shape[1]
        
        # Create FAISS index
        if index_type == 'flat':
            # Flat L2 index (exact search)
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif index_type == 'ivf':
            # IVF index (approximate search)
            nlist = min(100, len(documents) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            
            # Train index
            logger.info("Training IVF index...")
            self.index.train(self.embeddings.astype('float32'))
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add vectors to index
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of documents with scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.encode_texts(
            [query],
            method='sentence_transformer'
        )
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    'document': self.documents[idx],
                    'score': float(1 / (1 + dist)),  # Convert distance to similarity
                    'rank': i + 1
                }
                results.append(result)
        
        return results
    
    def save_index(self, path: Optional[Path] = None):
        """Save FAISS index and documents"""
        if path is None:
            path = FAISS_DIR
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / 'index.faiss'))
        
        # Save documents and embeddings
        with open(path / 'documents.pkl', 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'embedding_dim': self.embedding_dim
            }, f)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load_index(self, path: Optional[Path] = None):
        """Load FAISS index and documents"""
        if path is None:
            path = FAISS_DIR
        
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / 'index.faiss'))
        
        # Load documents and embeddings
        with open(path / 'documents.pkl', 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.embedding_dim = data['embedding_dim']
        
        logger.info(f"Loaded FAISS index from {path}")


def main():
    """Build and save FAISS index"""
    # Load corpus
    corpus_path = CORPUS_DIR / 'corpus.json'
    
    if not corpus_path.exists():
        print("Corpus not found. Please run build_corpus.py first.")
        return
    
    with open(corpus_path, 'r') as f:
        documents = json.load(f)
    
    # Build index
    builder = FAISSIndexBuilder()
    builder.build_index(documents, index_type='flat')
    
    # Save index
    builder.save_index()
    
    print(f"\nBuilt FAISS index with {len(documents)} documents")
    
    # Test search
    print("\nTesting search with query: 'What are the top performing stores?'")
    results = builder.search("What are the top performing stores?", top_k=3)
    
    for result in results:
        print(f"\nRank {result['rank']} (Score: {result['score']:.3f}):")
        print(result['document']['text'][:200] + "...")


if __name__ == "__main__":
    main()
