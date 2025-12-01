# Security Guardrails Documentation

## Overview

The Revenue Intelligence Platform implements comprehensive security guardrails to ensure reliability, security, and proper usage.

## Features

### 1. Input Validation ✅
- **Store ID**: Must be between 1-100
- **Department ID**: Must be between 1-100
- **Forecast Horizon**: Must be between 1-52 weeks
- **Model Type**: Must be 'lstm' or 'transformer'
- **Queries**: Maximum 500 characters, sanitized for SQL injection and XSS

### 2. Rate Limiting ⏱️
- **Forecast Endpoint**: 50 requests per minute per IP
- **RAG Endpoint**: 30 requests per minute per IP
- **General Endpoints**: 100 requests per minute per IP

Rate limit headers included in responses:
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-Response-Time`: Request processing time

### 3. Security Headers 🔒
All responses include:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000`
- `Content-Security-Policy`: Restricts resource loading

### 4. Request Sanitization 🧹
- Removes potentially harmful SQL patterns
- Filters XSS attempts
- Sanitizes special characters
- Validates JSON structure

### 5. Content Filtering 🛡️
Blocks requests containing:
- Script tags
- JavaScript event handlers
- Malware/exploit keywords
- Injection attempts

### 6. Error Handling 🚨
- Centralized error handling
- Detailed logging
- User-friendly error messages
- No sensitive information leakage

### 7. Monitoring & Logging 📊
- Request count by endpoint
- Error tracking
- Slow request detection (>5 seconds)
- Performance metrics

### 8. Circuit Breaker 🔌
- Protects external services (Neo4j, OpenAI)
- Automatic failure detection
- Graceful degradation
- Auto-recovery

## Usage Examples

### Valid Request
```python
POST /forecast
{
  "store_id": 1,
  "dept_id": 5,
  "horizon": 4,
  "model_type": "lstm",
  "natural_language_query": "Explain the forecast"
}
```

### Invalid Requests (Will Be Rejected)

#### Invalid Store ID
```python
{
  "store_id": 999,  # Out of range
  "dept_id": 1
}
# Response: 400 - "Store ID must be between 1 and 100"
```

#### SQL Injection Attempt
```python
{
  "natural_language_query": "'; DROP TABLE stores; --"
}
# Response: 400 - "Query contains potentially harmful SQL patterns"
```

#### XSS Attempt
```python
{
  "natural_language_query": "<script>alert('xss')</script>"
}
# Response: 400 - "Content Blocked"
```

#### Rate Limit Exceeded
```python
# After 50 forecast requests in 1 minute
# Response: 429 - "Rate limit exceeded. Please try again later."
```

## Monitoring

### Get System Stats
```python
GET /monitoring/stats
```

Response:
```json
{
  "total_requests": 1250,
  "requests_by_endpoint": {
    "/forecast": 450,
    "/rag-answer": 300,
    "/graph-insights": 500
  },
  "total_errors": 15,
  "errors_by_endpoint": {
    "/forecast": 5,
    "/rag-answer": 10
  },
  "slow_requests_count": 3
}
```

## Configuration

### Adjusting Rate Limits

Edit `utils/guardrails.py`:
```python
forecast_limiter = RateLimiter(max_requests=50, window_seconds=60)
rag_limiter = RateLimiter(max_requests=30, window_seconds=60)
```

### Adjusting Validation Ranges

Edit `utils/guardrails.py`:
```python
class InputValidator:
    MIN_STORE_ID = 1
    MAX_STORE_ID = 100
    MIN_HORIZON = 1
    MAX_HORIZON = 52
```

## Best Practices

1. **Always validate input** before processing
2. **Monitor rate limits** in your client applications
3. **Handle 429 errors** with exponential backoff
4. **Log all errors** for debugging
5. **Keep queries concise** (under 500 characters)
6. **Use HTTPS** in production

## Security Checklist

- ✅ Input validation on all endpoints
- ✅ Rate limiting per IP address
- ✅ SQL injection prevention
- ✅ XSS protection
- ✅ CORS properly configured
- ✅ Security headers enabled
- ✅ Error messages sanitized
- ✅ Request/response logging
- ✅ Circuit breaker for external services
- ✅ Content filtering

## Troubleshooting

### Rate Limit Errors
**Problem**: Getting 429 errors  
**Solution**: Wait 60 seconds or reduce request frequency

### Validation Errors
**Problem**: Getting 400 errors  
**Solution**: Check input ranges and query format

### Content Blocked
**Problem**: Request rejected as harmful  
**Solution**: Remove special characters and script tags

### Slow Requests
**Problem**: Requests taking >5 seconds  
**Solution**: Check system resources and database connections

## Production Recommendations

1. **Enable HTTPS**: Use SSL/TLS certificates
2. **Add Authentication**: Implement API keys or OAuth
3. **Increase Rate Limits**: Based on expected traffic
4. **Set up Monitoring**: Use Prometheus/Grafana
5. **Configure Alerts**: For errors and slow requests
6. **Regular Security Audits**: Review logs and patterns
7. **Update Dependencies**: Keep packages up-to-date
8. **Backup Data**: Regular database backups
9. **Load Testing**: Test under high traffic
10. **DDoS Protection**: Use CloudFlare or similar

## Support

For issues or questions about guardrails:
1. Check logs in `logs/api.log`
2. Review monitoring stats at `/monitoring/stats`
3. Verify configuration in `utils/guardrails.py`
4. Test with example requests above
