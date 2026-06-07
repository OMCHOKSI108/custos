"""
auth.py - API Key Authentication

OPTIONAL API KEY PROTECTION:
  If CUSTOS_API_KEY is set in environment → requests to protected endpoints
  must include a matching X-API-Key header.
  
  If CUSTOS_API_KEY is empty/unset → auth is disabled (demo mode).

PROTECTED ENDPOINTS:
  POST /chat, POST /train, POST /budget/configure, DELETE /cache, GET /export/csv

UNPROTECTED ENDPOINTS:
  GET /, /health, /docs, /stats, /predict, /compare, /history
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.config import CUSTOS_API_KEY

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)):
    """
    FastAPI dependency for protected endpoints.
    
    If CUSTOS_API_KEY is not set → allows all requests (demo mode).
    If CUSTOS_API_KEY is set → requires matching X-API-Key header.
    """
    # No key configured = auth disabled (backward compatible)
    if not CUSTOS_API_KEY:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_api_key",
                "message": "X-API-Key header is required. Set CUSTOS_API_KEY in your environment.",
            },
        )

    if api_key != CUSTOS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "invalid_api_key",
                "message": "Invalid API key.",
            },
        )

    return api_key
