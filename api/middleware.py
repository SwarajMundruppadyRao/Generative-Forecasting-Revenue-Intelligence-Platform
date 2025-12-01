"""
Middleware for the Revenue Intelligence Platform API
"""
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import time
import logging
from utils.guardrails import (
    SECURITY_HEADERS,
    request_monitor,
    ContentFilter,
    RequestSanitizer
)

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response


class RequestMonitoringMiddleware(BaseHTTPMiddleware):
    """Monitor and log all requests"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log metrics
            request_monitor.log_request(
                endpoint=request.url.path,
                duration=duration,
                status=response.status_code
            )
            
            # Add timing header
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
        
        except Exception as e:
            duration = time.time() - start_time
            request_monitor.log_request(
                endpoint=request.url.path,
                duration=duration,
                status=500
            )
            logger.error(f"Request failed: {e}")
            raise


class ContentFilterMiddleware(BaseHTTPMiddleware):
    """Filter potentially harmful content"""
    
    async def dispatch(self, request: Request, call_next):
        # Check request body for harmful content
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    body_str = body.decode('utf-8')
                    if not ContentFilter.is_safe(body_str):
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": "Content Blocked",
                                "message": "Request contains potentially harmful content"
                            }
                        )
            except Exception as e:
                logger.warning(f"Content filter error: {e}")
        
        response = await call_next(request)
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Validation Error",
                    "message": str(e),
                    "type": "validation_error"
                }
            )
        
        except PermissionError as e:
            logger.warning(f"Permission error: {e}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Permission Denied",
                    "message": "You don't have permission to access this resource",
                    "type": "permission_error"
                }
            )
        
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred. Please try again later.",
                    "type": "server_error"
                }
            )


def setup_cors(app):
    """Setup CORS middleware"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Response-Time", "X-RateLimit-Remaining"]
    )


def setup_middleware(app):
    """Setup all middleware"""
    # Order matters - add in reverse order of execution
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(ContentFilterMiddleware)
    app.add_middleware(RequestMonitoringMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    setup_cors(app)
