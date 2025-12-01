"""
Guardrails and validation utilities for the Revenue Intelligence Platform
"""
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import re
from functools import wraps
from fastapi import HTTPException, Request
from pydantic import BaseModel, validator, Field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class InputValidator:
    """Validates and sanitizes user inputs"""
    
    # Allowed ranges
    MIN_STORE_ID = 1
    MAX_STORE_ID = 100
    MIN_DEPT_ID = 1
    MAX_DEPT_ID = 100
    MIN_HORIZON = 1
    MAX_HORIZON = 52  # Max 1 year forecast
    MAX_QUERY_LENGTH = 500
    
    @staticmethod
    def validate_store_id(store_id: int) -> int:
        """Validate store ID"""
        if not isinstance(store_id, int):
            raise ValueError("Store ID must be an integer")
        if store_id < InputValidator.MIN_STORE_ID or store_id > InputValidator.MAX_STORE_ID:
            raise ValueError(f"Store ID must be between {InputValidator.MIN_STORE_ID} and {InputValidator.MAX_STORE_ID}")
        return store_id
    
    @staticmethod
    def validate_dept_id(dept_id: int) -> int:
        """Validate department ID"""
        if not isinstance(dept_id, int):
            raise ValueError("Department ID must be an integer")
        if dept_id < InputValidator.MIN_DEPT_ID or dept_id > InputValidator.MAX_DEPT_ID:
            raise ValueError(f"Department ID must be between {InputValidator.MIN_DEPT_ID} and {InputValidator.MAX_DEPT_ID}")
        return dept_id
    
    @staticmethod
    def validate_horizon(horizon: int) -> int:
        """Validate forecast horizon"""
        if not isinstance(horizon, int):
            raise ValueError("Horizon must be an integer")
        if horizon < InputValidator.MIN_HORIZON or horizon > InputValidator.MAX_HORIZON:
            raise ValueError(f"Horizon must be between {InputValidator.MIN_HORIZON} and {InputValidator.MAX_HORIZON}")
        return horizon
    
    @staticmethod
    def validate_model_type(model_type: str) -> str:
        """Validate model type"""
        allowed_models = ['lstm', 'transformer']
        model_type = model_type.lower().strip()
        if model_type not in allowed_models:
            raise ValueError(f"Model type must be one of: {', '.join(allowed_models)}")
        return model_type
    
    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitize natural language query"""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        # Check length
        if len(query) > InputValidator.MAX_QUERY_LENGTH:
            raise ValueError(f"Query too long. Maximum {InputValidator.MAX_QUERY_LENGTH} characters")
        
        # Remove potentially harmful characters
        query = re.sub(r'[<>{}[\]\\]', '', query)
        
        # Check for SQL injection patterns
        sql_patterns = [
            r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE)\b',
            r'--',
            r'/\*.*\*/',
            r';.*--'
        ]
        for pattern in sql_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError("Query contains potentially harmful SQL patterns")
        
        return query.strip()
    
    @staticmethod
    def validate_pagination(skip: int, limit: int) -> tuple:
        """Validate pagination parameters"""
        if skip < 0:
            raise ValueError("Skip must be non-negative")
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        return skip, limit


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        return max(0, self.max_requests - len(self.requests[client_id]))


# Global rate limiter instances
forecast_limiter = RateLimiter(max_requests=50, window_seconds=60)  # 50 forecasts per minute
rag_limiter = RateLimiter(max_requests=30, window_seconds=60)  # 30 RAG queries per minute
general_limiter = RateLimiter(max_requests=100, window_seconds=60)  # 100 general requests per minute


def rate_limit(limiter: RateLimiter):
    """Decorator for rate limiting"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_id = request.client.host
            
            if not limiter.is_allowed(client_id):
                remaining = limiter.get_remaining(client_id)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Please try again later.",
                        "remaining": remaining,
                        "reset_in_seconds": limiter.window_seconds
                    }
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# ERROR HANDLING
# ============================================================================

class ErrorHandler:
    """Centralized error handling"""
    
    @staticmethod
    def handle_validation_error(error: Exception) -> Dict[str, Any]:
        """Handle validation errors"""
        return {
            "error": "Validation Error",
            "message": str(error),
            "type": "validation_error"
        }
    
    @staticmethod
    def handle_model_error(error: Exception) -> Dict[str, Any]:
        """Handle model errors"""
        logger.error(f"Model error: {error}")
        return {
            "error": "Model Error",
            "message": "An error occurred during prediction. Please try again.",
            "type": "model_error"
        }
    
    @staticmethod
    def handle_data_error(error: Exception) -> Dict[str, Any]:
        """Handle data errors"""
        logger.error(f"Data error: {error}")
        return {
            "error": "Data Error",
            "message": "Unable to process data. Please check your input.",
            "type": "data_error"
        }
    
    @staticmethod
    def handle_api_error(error: Exception) -> Dict[str, Any]:
        """Handle API errors"""
        logger.error(f"API error: {error}")
        return {
            "error": "API Error",
            "message": "An error occurred while processing your request.",
            "type": "api_error"
        }


# ============================================================================
# SECURITY HEADERS
# ============================================================================

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com;",
}


# ============================================================================
# REQUEST SANITIZATION
# ============================================================================

class RequestSanitizer:
    """Sanitizes incoming requests"""
    
    @staticmethod
    def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize request headers"""
        # Remove potentially harmful headers
        dangerous_headers = ['X-Forwarded-Host', 'X-Original-URL']
        return {k: v for k, v in headers.items() if k not in dangerous_headers}
    
    @staticmethod
    def sanitize_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize JSON data"""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            key = re.sub(r'[^\w\s-]', '', str(key))
            
            # Sanitize value
            if isinstance(value, str):
                value = InputValidator.sanitize_query(value)
            elif isinstance(value, dict):
                value = RequestSanitizer.sanitize_json(value)
            elif isinstance(value, list):
                value = [RequestSanitizer.sanitize_json(item) if isinstance(item, dict) else item for item in value]
            
            sanitized[key] = value
        
        return sanitized


# ============================================================================
# MONITORING & LOGGING
# ============================================================================

class RequestMonitor:
    """Monitors and logs requests"""
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.error_count = defaultdict(int)
        self.slow_requests = []
    
    def log_request(self, endpoint: str, duration: float, status: int):
        """Log request metrics"""
        self.request_count[endpoint] += 1
        
        if status >= 400:
            self.error_count[endpoint] += 1
        
        if duration > 5.0:  # Log slow requests (>5 seconds)
            self.slow_requests.append({
                'endpoint': endpoint,
                'duration': duration,
                'timestamp': datetime.now()
            })
            logger.warning(f"Slow request: {endpoint} took {duration:.2f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            'total_requests': sum(self.request_count.values()),
            'requests_by_endpoint': dict(self.request_count),
            'total_errors': sum(self.error_count.values()),
            'errors_by_endpoint': dict(self.error_count),
            'slow_requests_count': len(self.slow_requests)
        }


# Global monitor
request_monitor = RequestMonitor()


# ============================================================================
# CONTENT FILTERING
# ============================================================================

class ContentFilter:
    """Filters inappropriate or harmful content"""
    
    # Blocked patterns
    BLOCKED_PATTERNS = [
        r'\b(hack|exploit|inject|malware|virus)\b',
        r'<script.*?>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',  # Event handlers
    ]
    
    @staticmethod
    def is_safe(content: str) -> bool:
        """Check if content is safe"""
        content_lower = content.lower()
        
        for pattern in ContentFilter.BLOCKED_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                logger.warning(f"Blocked content matching pattern: {pattern}")
                return False
        
        return True
    
    @staticmethod
    def filter_response(response: str, max_length: int = 5000) -> str:
        """Filter and truncate response"""
        # Remove any script tags
        response = re.sub(r'<script.*?>.*?</script>', '', response, flags=re.IGNORECASE | re.DOTALL)
        
        # Truncate if too long
        if len(response) > max_length:
            response = response[:max_length] + "... [truncated]"
        
        return response


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """Circuit breaker pattern for external services"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = defaultdict(int)
        self.last_failure_time = defaultdict(lambda: None)
        self.state = defaultdict(lambda: 'closed')  # closed, open, half-open
    
    def call(self, service_name: str, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        # Check if circuit is open
        if self.state[service_name] == 'open':
            if self.last_failure_time[service_name]:
                time_since_failure = (datetime.now() - self.last_failure_time[service_name]).seconds
                if time_since_failure > self.timeout:
                    self.state[service_name] = 'half-open'
                else:
                    raise Exception(f"Circuit breaker open for {service_name}")
        
        try:
            result = func(*args, **kwargs)
            
            # Reset on success
            if self.state[service_name] == 'half-open':
                self.state[service_name] = 'closed'
                self.failures[service_name] = 0
            
            return result
        
        except Exception as e:
            self.failures[service_name] += 1
            self.last_failure_time[service_name] = datetime.now()
            
            if self.failures[service_name] >= self.failure_threshold:
                self.state[service_name] = 'open'
                logger.error(f"Circuit breaker opened for {service_name}")
            
            raise e


# Global circuit breaker
circuit_breaker = CircuitBreaker()
