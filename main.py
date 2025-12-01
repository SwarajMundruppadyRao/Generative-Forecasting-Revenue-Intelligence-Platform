"""
Main entry point for the Revenue Intelligence Platform
Run preprocessing, training, and setup in one command
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.preprocess import DataPreprocessor
from forecasting.train import train_lstm_model, train_transformer_model
from forecasting.dataset import split_train_val
from graph.build_graph import GraphBuilder
from knowledge_graph.neo4j_loader import Neo4jLoader
from rag.build_corpus import CorpusBuilder
from rag.build_faiss import FAISSIndexBuilder
from rag.hybrid_retrieval import HybridRetriever
from utils.config import BASE_DIR, TRAIN_FILE, STORES_FILE
from utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__, BASE_DIR / "logs" / "main.log")


def setup_data():
    """Preprocess data"""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preprocessing")
    logger.info("=" * 60)
    
    preprocessor = DataPreprocessor()
    train, test = preprocessor.preprocess_pipeline(
        save_path=BASE_DIR / 'data' / 'processed'
    )
    
    logger.info(f"✓ Preprocessed data saved")
    return train, test


def setup_graph():
    """Build graph structure"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Graph Construction")
    logger.info("=" * 60)
    
    train_data = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'train_processed.csv')
    stores_data = pd.read_csv(STORES_FILE)
    
    builder = GraphBuilder()
    graph = builder.build_from_data(train_data, stores_data)
    builder.save_graph(graph, BASE_DIR / 'graph' / 'sales_graph.pt')
    
    logger.info("✓ Graph built and saved")


def setup_neo4j():
    """Load Neo4j knowledge graph"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Neo4j Knowledge Graph")
    logger.info("=" * 60)
    
    train_data = pd.read_csv(TRAIN_FILE)
    stores_data = pd.read_csv(STORES_FILE)
    
    # Sample for faster loading
    train_data = train_data.sample(min(10000, len(train_data)), random_state=42)
    
    loader = Neo4jLoader()
    
    if loader.driver:
        loader.load_all_data(train_data, stores_data)
        stats = loader.get_stats()
        logger.info(f"✓ Neo4j loaded: {stats}")
        loader.close()
    else:
        logger.warning("⚠ Neo4j not available - skipping")


def setup_rag():
    """Build RAG corpus and indices"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: RAG Pipeline Setup")
    logger.info("=" * 60)
    
    # Load data
    train_data = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'train_processed.csv')
    
    # Build corpus
    logger.info("Building corpus...")
    corpus_builder = CorpusBuilder()
    documents = corpus_builder.build_corpus(train_data)
    corpus_builder.save_corpus()
    logger.info(f"✓ Corpus built: {len(documents)} documents")
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    faiss_builder = FAISSIndexBuilder()
    faiss_builder.build_index(documents, index_type='flat')
    faiss_builder.save_index()
    logger.info("✓ FAISS index built")
    
    # Build hybrid retrieval
    logger.info("Building hybrid retrieval...")
    retriever = HybridRetriever()
    retriever.load_dense_index()
    retriever.build_sparse_index()
    retriever.save_sparse_index()
    logger.info("✓ Hybrid retrieval ready")


def train_models():
    """Train forecasting models"""
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Model Training")
    logger.info("=" * 60)
    
    # Load data
    train_data = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'train_processed.csv')
    train_data, val_data = split_train_val(train_data, val_split=0.2)
    
    # Train LSTM
    logger.info("\nTraining LSTM model...")
    train_lstm_model(train_data, val_data)
    logger.info("✓ LSTM model trained")
    
    # Train Transformer
    logger.info("\nTraining Transformer model...")
    train_transformer_model(train_data, val_data)
    logger.info("✓ Transformer model trained")


def main():
    """Main setup pipeline"""
    parser = argparse.ArgumentParser(
        description="Revenue Intelligence Platform Setup"
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['data', 'graph', 'neo4j', 'rag', 'train', 'all'],
        default=['all'],
        help='Steps to run'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (faster setup)'
    )
    
    args = parser.parse_args()
    
    steps = args.steps
    if 'all' in steps:
        steps = ['data', 'graph', 'neo4j', 'rag', 'train']
    
    if args.skip_training and 'train' in steps:
        steps.remove('train')
    
    logger.info("🚀 Starting Revenue Intelligence Platform Setup")
    logger.info(f"Steps to run: {', '.join(steps)}")
    
    try:
        if 'data' in steps:
            setup_data()
        
        if 'graph' in steps:
            setup_graph()
        
        if 'neo4j' in steps:
            setup_neo4j()
        
        if 'rag' in steps:
            setup_rag()
        
        if 'train' in steps:
            train_models()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ SETUP COMPLETE!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Start the API server:")
        logger.info("   python -m api.server")
        logger.info("\n2. Or use Docker:")
        logger.info("   cd docker && docker-compose up -d")
        logger.info("\n3. Visit API docs:")
        logger.info("   http://localhost:8000/docs")
        
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        raise


if __name__ == "__main__":
    main()
