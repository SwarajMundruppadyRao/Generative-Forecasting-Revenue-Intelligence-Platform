"""
API schemas for request/response models with guardrails
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from utils.guardrails import InputValidator


class ForecastRequest(BaseModel):
    """Request model for forecast endpoint"""
    store_id: int = Field(..., description="Store ID", ge=1, le=100)
    dept_id: Optional[int] = Field(None, description="Department ID (optional)", ge=1, le=100)
    horizon: int = Field(4, description="Forecast horizon in weeks", ge=1, le=52)
    model_type: str = Field("lstm", description="Model type: 'lstm' or 'transformer'")
    natural_language_query: Optional[str] = Field(None, description="Optional natural language query", max_length=500)
    
    @validator('store_id')
    def validate_store(cls, v):
        return InputValidator.validate_store_id(v)
    
    @validator('dept_id')
    def validate_dept(cls, v):
        if v is not None:
            return InputValidator.validate_dept_id(v)
        return v
    
    @validator('horizon')
    def validate_horizon_value(cls, v):
        return InputValidator.validate_horizon(v)
    
    @validator('model_type')
    def validate_model(cls, v):
        return InputValidator.validate_model_type(v)
    
    @validator('natural_language_query')
    def validate_query(cls, v):
        if v is not None:
            return InputValidator.sanitize_query(v)
        return v


class ForecastResponse(BaseModel):
    """Response model for forecast endpoint"""
    store_id: int
    dept_id: Optional[int]
    predictions: List[float]
    forecast_dates: List[str]
    model_type: str
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class RAGRequest(BaseModel):
    """Request model for RAG answer endpoint"""
    question: str = Field(..., description="Natural language question", max_length=500)
    store_id: Optional[int] = Field(None, description="Optional store ID for context", ge=1, le=100)
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    
    @validator('question')
    def validate_question(cls, v):
        return InputValidator.sanitize_query(v)
    
    @validator('store_id')
    def validate_store(cls, v):
        if v is not None:
            return InputValidator.validate_store_id(v)
        return v


class RAGResponse(BaseModel):
    """Response model for RAG answer endpoint"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class GraphInsightsRequest(BaseModel):
    """Request model for graph insights endpoint"""
    store_id: int = Field(..., description="Store ID", ge=1, le=100)
    include_similar_stores: bool = Field(True, description="Include similar stores")
    include_departments: bool = Field(True, description="Include department information")
    
    @validator('store_id')
    def validate_store(cls, v):
        return InputValidator.validate_store_id(v)


class GraphInsightsResponse(BaseModel):
    """Response model for graph insights endpoint"""
    store_id: int
    store_info: Dict[str, Any]
    departments: Optional[List[Dict[str, Any]]] = None
    similar_stores: Optional[List[Dict[str, Any]]] = None
    gnn_embeddings: Optional[List[float]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    services: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str
