from fastapi import Header, HTTPException
from app.config import settings

def require_api_key(x_api_key: str | None = Header(default=None)):
    """
    If settings.api_key is set, require clients to send X-API-Key matching it.
    If settings.api_key is None/empty, allow requests (dev mode).
    """
    if not settings.api_key:
        return  # dev mode: no auth required

    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
