"""
RAG pipeline using LangChain and OpenAI
"""
from typing import List, Dict, Optional
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import openai

from rag.hybrid_retrieval import HybridRetriever
from knowledge_graph.neo4j_query import Neo4jQuery
from utils.config import OPENAI_CONFIG, RAG_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """RAG pipeline for question answering"""
    
    def __init__(
        self,
        use_neo4j: bool = True,
        use_hybrid_retrieval: bool = True
    ):
        """
        Initialize RAG pipeline
        
        Args:
            use_neo4j: Whether to use Neo4j for additional context
            use_hybrid_retrieval: Whether to use hybrid retrieval
        """
        self.use_neo4j = use_neo4j
        self.use_hybrid_retrieval = use_hybrid_retrieval
        
        # Initialize OpenAI
        openai.api_key = OPENAI_CONFIG['api_key']
        
        try:
            self.llm = ChatOpenAI(
                model=OPENAI_CONFIG['model'],
                temperature=OPENAI_CONFIG['temperature'],
                max_tokens=OPENAI_CONFIG['max_tokens'],
                api_key=OPENAI_CONFIG['api_key']
            )
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI: {e}")
            self.llm = None
        
        # Initialize retriever
        if use_hybrid_retrieval:
            self.retriever = HybridRetriever(
                dense_weight=RAG_CONFIG['dense_weight'],
                sparse_weight=RAG_CONFIG['sparse_weight'],
                gnn_weight=RAG_CONFIG['gnn_weight']
            )
            
            try:
                self.retriever.load_dense_index()
                self.retriever.load_sparse_index()
            except Exception as e:
                logger.warning(f"Failed to load retrieval indices: {e}")
        
        # Initialize Neo4j
        if use_neo4j:
            try:
                self.neo4j_query = Neo4jQuery()
            except Exception as e:
                logger.warning(f"Failed to initialize Neo4j: {e}")
                self.neo4j_query = None
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        store_id: Optional[int] = None
    ) -> Dict:
        """
        Retrieve relevant context for query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            store_id: Optional store ID for Neo4j context
        
        Returns:
            Dictionary with retrieved context
        """
        context = {
            'documents': [],
            'neo4j_context': '',
            'query': query
        }
        
        # Retrieve documents
        if self.use_hybrid_retrieval:
            try:
                results = self.retriever.hybrid_search(query, top_k=top_k)
                context['documents'] = results
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
        
        # Get Neo4j context
        if self.use_neo4j and store_id and self.neo4j_query and self.neo4j_query.driver:
            try:
                neo4j_context = self.neo4j_query.get_revenue_context(store_id)
                context['neo4j_context'] = neo4j_context
            except Exception as e:
                logger.error(f"Neo4j error: {e}")
        
        return context
    
    def generate_answer(
        self,
        query: str,
        context: Dict,
        include_explanation: bool = True
    ) -> Dict:
        """
        Generate answer using LLM
        
        Args:
            query: User query
            context: Retrieved context
            include_explanation: Whether to include explanation
        
        Returns:
            Dictionary with answer and metadata
        """
        if not self.llm:
            return {
                'answer': 'LLM not available. Please configure OpenAI API key.',
                'sources': [],
                'error': 'LLM not initialized'
            }
        
        # Prepare context text
        context_text = ""
        
        if context['documents']:
            context_text += "Relevant Information:\n\n"
            for i, doc in enumerate(context['documents'][:3]):
                context_text += f"Document {i+1}:\n{doc['document']['text']}\n\n"
        
        if context['neo4j_context']:
            context_text += f"Knowledge Graph Context:\n{context['neo4j_context']}\n\n"
        
        # Create prompt
        system_message = """You are an expert retail analytics assistant specializing in revenue forecasting and sales analysis. 
Your role is to provide accurate, data-driven insights based on the provided context.

When answering:
1. Base your response on the provided context
2. Be specific with numbers and statistics
3. Provide clear explanations
4. If the context doesn't contain enough information, acknowledge this
5. Format your response clearly with bullet points where appropriate"""
        
        user_message = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        try:
            # Generate response
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Prepare sources
            sources = []
            for doc in context['documents'][:3]:
                source = {
                    'text': doc['document']['text'][:200] + "...",
                    'metadata': doc['document'].get('metadata', {}),
                    'score': doc['score']
                }
                sources.append(source)
            
            return {
                'answer': answer,
                'sources': sources,
                'query': query
            }
        
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return {
                'answer': f'Error generating answer: {str(e)}',
                'sources': [],
                'error': str(e)
            }
    
    def answer_question(
        self,
        query: str,
        store_id: Optional[int] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Complete RAG pipeline: retrieve and generate
        
        Args:
            query: User question
            store_id: Optional store ID
            top_k: Number of documents to retrieve
        
        Returns:
            Answer with sources
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve context
        context = self.retrieve_context(query, top_k=top_k, store_id=store_id)
        
        # Generate answer
        answer = self.generate_answer(query, context)
        
        return answer
    
    def close(self):
        """Close connections"""
        if self.neo4j_query:
            self.neo4j_query.close()


def main():
    """Test RAG pipeline"""
    # Initialize pipeline
    pipeline = RAGPipeline(use_neo4j=True, use_hybrid_retrieval=True)
    
    # Test questions
    questions = [
        "What are the top performing stores?",
        "How do holiday sales compare to regular sales?",
        "What departments have the highest sales?",
        "What are the sales trends over time?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)
        
        result = pipeline.answer_question(question, store_id=1)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        if result.get('sources'):
            print(f"\nSources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources']):
                print(f"  {i+1}. Score: {source['score']:.3f}")
                print(f"     {source['text'][:100]}...")
        
        print()
    
    pipeline.close()


if __name__ == "__main__":
    main()
