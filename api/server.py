"""
FastAPI server for Revenue Intelligence Platform with Guardrails
"""
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

from api.schemas import (
    ForecastRequest, ForecastResponse,
    RAGRequest, RAGResponse,
    GraphInsightsRequest, GraphInsightsResponse,
    HealthResponse, ErrorResponse
)
from api.middleware import setup_middleware
from forecasting.predict import ForecastingPredictor
from rag.rag_pipeline import RAGPipeline
from knowledge_graph.neo4j_query import Neo4jQuery
from graph.gnn_model import HeteroGNN, create_gnn_model
from graph.build_graph import GraphBuilder
from utils.config import API_CONFIG, BASE_DIR, MODELS_DIR
from utils.logger import setup_logger
from utils.guardrails import (
    forecast_limiter,
    rag_limiter,
    general_limiter,
    rate_limit,
    request_monitor
)

# Setup logging
logger = setup_logger(__name__, BASE_DIR / "logs" / "api.log")

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG['title'],
    version=API_CONFIG['version'],
    description="Production-grade Revenue Intelligence Platform with forecasting, RAG, and graph analytics"
)

# Setup all middleware (CORS, security headers, monitoring, etc.)
setup_middleware(app)

# Include monitoring router
from api.monitoring import router as monitoring_router
app.include_router(monitoring_router)

# Global state for models and data
class AppState:
    def __init__(self):
        self.lstm_predictor = None
        self.transformer_predictor = None
        self.rag_pipeline = None
        self.neo4j_query = None
        self.gnn_model = None
        self.graph = None
        self.data = None
        self.initialized = False

state = AppState()


@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup"""
    logger.info("Starting up Revenue Intelligence Platform API...")
    
    try:
        # Load data
        processed_data_path = BASE_DIR / 'data' / 'processed' / 'train_processed.csv'
        if processed_data_path.exists():
            state.data = pd.read_csv(processed_data_path)
            logger.info(f"Loaded data: {state.data.shape}")
        else:
            logger.warning("Processed data not found")
        
        # Initialize forecasting models
        try:
            state.lstm_predictor = ForecastingPredictor(model_type="lstm")
            logger.info("Loaded LSTM forecaster")
        except Exception as e:
            logger.warning(f"Failed to load LSTM model: {e}")
        
        try:
            state.transformer_predictor = ForecastingPredictor(model_type="transformer")
            logger.info("Loaded Transformer forecaster")
        except Exception as e:
            logger.warning(f"Failed to load Transformer model: {e}")
        
        # Initialize RAG pipeline
        try:
            state.rag_pipeline = RAGPipeline(use_neo4j=True, use_hybrid_retrieval=True)
            logger.info("Initialized RAG pipeline")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG pipeline: {e}")
        
        # Initialize Neo4j
        try:
            state.neo4j_query = Neo4jQuery()
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j: {e}")
        
        # Load GNN model and graph
        try:
            graph_path = BASE_DIR / 'graph' / 'sales_graph.pt'
            if graph_path.exists():
                builder = GraphBuilder()
                state.graph = builder.load_graph(graph_path)
                state.gnn_model = create_gnn_model(state.graph.metadata())
                
                # Load trained weights if available
                gnn_weights_path = MODELS_DIR / 'gnn_model.pth'
                if gnn_weights_path.exists():
                    import torch
                    checkpoint = torch.load(gnn_weights_path, map_location='cpu', weights_only=False)
                    state.gnn_model.load_state_dict(checkpoint['model_state_dict'])
                
                logger.info("Loaded GNN model and graph")
        except Exception as e:
            logger.warning(f"Failed to load GNN: {e}")
        
        state.initialized = True
        logger.info("API startup complete!")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    
    if state.rag_pipeline:
        state.rag_pipeline.close()
    
    if state.neo4j_query:
        state.neo4j_query.close()


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check"""
    return {
        "status": "healthy" if state.initialized else "initializing",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "lstm_forecaster": state.lstm_predictor is not None,
            "transformer_forecaster": state.transformer_predictor is not None,
            "rag_pipeline": state.rag_pipeline is not None,
            "neo4j": state.neo4j_query is not None and state.neo4j_query.driver is not None,
            "gnn": state.gnn_model is not None
        }
    }


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Generate revenue forecast for a store/department
    
    This endpoint combines:
    1. Time series forecasting (LSTM or Transformer)
    2. Optional RAG-based explanation
    3. Neo4j context for insights
    """
    try:
        logger.info(f"Forecast request: Store {request.store_id}, Dept {request.dept_id}")
        
        if state.data is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Data not loaded"
            )
        
        # Select predictor
        if request.model_type == "lstm":
            predictor = state.lstm_predictor
        elif request.model_type == "transformer":
            predictor = state.transformer_predictor
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type: {request.model_type}"
            )
        
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{request.model_type} model not available"
            )
        
        # Make prediction
        result = predictor.predict_store(
            state.data,
            store_id=request.store_id,
            dept_id=request.dept_id,
            horizon=request.horizon
        )
        
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result['error']
            )
        
        # Generate explanation if requested
        explanation = None
        if request.natural_language_query and state.rag_pipeline:
            try:
                # Enhance query with forecast context
                enhanced_query = f"{request.natural_language_query} For Store {request.store_id}"
                if request.dept_id:
                    enhanced_query += f" Department {request.dept_id}"
                
                rag_result = state.rag_pipeline.answer_question(
                    enhanced_query,
                    store_id=request.store_id
                )
                explanation = rag_result.get('answer', '')
            except Exception as e:
                logger.error(f"RAG explanation error: {e}")
        
        return ForecastResponse(
            store_id=result['store_id'],
            dept_id=result.get('dept_id'),
            predictions=result['predictions'],
            forecast_dates=result['forecast_dates'],
            model_type=result['model_type'],
            explanation=explanation,
            metadata={
                'last_date': result.get('last_date'),
                'horizon': request.horizon
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/rag-answer", response_model=RAGResponse)
async def rag_answer(request: RAGRequest):
    """
    Answer natural language questions using RAG
    
    This endpoint:
    1. Retrieves relevant documents using hybrid retrieval
    2. Optionally queries Neo4j for graph context
    3. Generates answer using LLM
    """
    try:
        logger.info(f"RAG request: {request.question}")
        
        if state.rag_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG pipeline not available"
            )
        
        # Answer question
        result = state.rag_pipeline.answer_question(
            request.question,
            store_id=request.store_id,
            top_k=request.top_k
        )
        
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['error']
            )
        
        return RAGResponse(
            question=request.question,
            answer=result['answer'],
            sources=result.get('sources', []),
            metadata={
                'store_id': request.store_id,
                'top_k': request.top_k
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG answer error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/graph-insights", response_model=GraphInsightsResponse)
async def graph_insights(request: GraphInsightsRequest):
    """
    Get graph-based insights for a store
    
    This endpoint:
    1. Queries Neo4j for store information
    2. Retrieves similar stores and departments
    3. Optionally provides GNN embeddings
    """
    try:
        logger.info(f"Graph insights request: Store {request.store_id}")
        
        if state.neo4j_query is None or state.neo4j_query.driver is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Neo4j not available"
            )
        
        # Get store insights
        insights = state.neo4j_query.get_store_insights(request.store_id)
        
        response_data = {
            'store_id': request.store_id,
            'store_info': insights.get('store_info', {}),
        }
        
        if request.include_departments:
            response_data['departments'] = insights.get('departments', [])
        
        if request.include_similar_stores:
            response_data['similar_stores'] = insights.get('similar_stores', [])
        
        # Get GNN embeddings if available
        if state.gnn_model and state.graph:
            try:
                import torch
                embeddings = state.gnn_model.get_embeddings(
                    state.graph.x_dict,
                    state.graph.edge_index_dict
                )
                
                # Get store embedding
                if 'store' in embeddings:
                    store_embeddings = embeddings['store'].detach().cpu().numpy()
                    # Find store index (simplified - would need proper mapping)
                    if request.store_id < len(store_embeddings):
                        response_data['gnn_embeddings'] = store_embeddings[request.store_id].tolist()
            except Exception as e:
                logger.warning(f"GNN embedding error: {e}")
        
        return GraphInsightsResponse(**response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph insights error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return await root()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.server:app",
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        reload=API_CONFIG['reload']
    )
