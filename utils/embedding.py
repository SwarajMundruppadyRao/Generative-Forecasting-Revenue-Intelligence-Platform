"""
Embedding utilities for text and time series
"""
import numpy as np
import torch
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import openai
from utils.config import OPENAI_CONFIG, RAG_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using multiple methods"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name or RAG_CONFIG["embedding_model"]
        self.sentence_model = None
        self.embedding_dim = RAG_CONFIG["embedding_dim"]
        
    def _load_sentence_transformer(self):
        """Lazy load sentence transformer"""
        if self.sentence_model is None:
            logger.info(f"Loading sentence transformer: {self.model_name}")
            self.sentence_model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
    
    def encode_texts(self, texts: List[str], method: str = "sentence_transformer") -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of text strings
            method: 'sentence_transformer' or 'openai'
        
        Returns:
            Embeddings array of shape (len(texts), embedding_dim)
        """
        if method == "sentence_transformer":
            self._load_sentence_transformer()
            embeddings = self.sentence_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        
        elif method == "openai":
            return self._encode_openai(texts)
        
        else:
            raise ValueError(f"Unknown embedding method: {method}")
    
    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """Encode using OpenAI embeddings"""
        openai.api_key = OPENAI_CONFIG["api_key"]
        embeddings = []
        
        for text in texts:
            try:
                response = openai.embeddings.create(
                    input=text,
                    model=OPENAI_CONFIG["embedding_model"]
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                logger.error(f"Error encoding text with OpenAI: {e}")
                # Fallback to zeros
                embeddings.append(np.zeros(1536))  # OpenAI embedding dim
        
        return np.array(embeddings)
    
    def encode_time_series(self, time_series: np.ndarray) -> np.ndarray:
        """
        Encode time series to fixed-size embeddings
        
        Args:
            time_series: Array of shape (n_samples, seq_length, n_features)
        
        Returns:
            Embeddings of shape (n_samples, embedding_dim)
        """
        # Simple statistical features as embeddings
        embeddings = []
        
        for ts in time_series:
            features = []
            # Mean, std, min, max
            features.extend([
                np.mean(ts, axis=0),
                np.std(ts, axis=0),
                np.min(ts, axis=0),
                np.max(ts, axis=0)
            ])
            # Trend (linear regression slope)
            if len(ts) > 1:
                x = np.arange(len(ts))
                for feat_idx in range(ts.shape[1]):
                    slope = np.polyfit(x, ts[:, feat_idx], 1)[0]
                    features.append([slope])
            
            embedding = np.concatenate([f.flatten() for f in features])
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Pad or truncate to fixed dimension
        if embeddings.shape[1] < self.embedding_dim:
            padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
            embeddings = np.concatenate([embeddings, padding], axis=1)
        else:
            embeddings = embeddings[:, :self.embedding_dim]
        
        return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def batch_cosine_similarity(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between queries and corpus
    
    Args:
        queries: Array of shape (n_queries, dim)
        corpus: Array of shape (n_corpus, dim)
    
    Returns:
        Similarity matrix of shape (n_queries, n_corpus)
    """
    # Normalize
    queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity
    similarity = np.dot(queries_norm, corpus_norm.T)
    return similarity
