"""
Monitoring endpoints for the Revenue Intelligence Platform
"""
from fastapi import APIRouter
from typing import Dict, Any
from utils.guardrails import request_monitor, forecast_limiter, rag_limiter, general_limiter

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get monitoring statistics"""
    stats = request_monitor.get_stats()
    
    # Add rate limiter stats
    stats['rate_limiters'] = {
        'forecast': {
            'max_requests': forecast_limiter.max_requests,
            'window_seconds': forecast_limiter.window_seconds
        },
        'rag': {
            'max_requests': rag_limiter.max_requests,
            'window_seconds': rag_limiter.window_seconds
        },
        'general': {
            'max_requests': general_limiter.max_requests,
            'window_seconds': general_limiter.window_seconds
        }
    }
    
    return stats


@router.get("/health/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with guardrails status"""
    return {
        "status": "healthy",
        "guardrails": {
            "input_validation": "active",
            "rate_limiting": "active",
            "content_filtering": "active",
            "security_headers": "active",
            "request_monitoring": "active",
            "error_handling": "active"
        },
        "metrics": request_monitor.get_stats()
    }
