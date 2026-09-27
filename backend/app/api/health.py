"""
Health check router for Sentinel API.
"""
from fastapi import APIRouter
try:
    from app.core.config import settings
except ImportError:
    from backend.app.core.config import settings

from ai.investigation.provider import get_llm_provider
from ai.investigation.local_provider import check_local_llm_health

router = APIRouter()


@router.get("/health", tags=["Health"])
def get_health():
    """
    Health check endpoint to verify backend service availability and LLM provider status.
    """
    prov_setting = getattr(settings, "LLM_PROVIDER", "local_ollama")
    display_name = "Local Ollama" if prov_setting == "local_ollama" else prov_setting.capitalize()

    try:
        provider = get_llm_provider()
        is_avail = provider.is_available()
    except Exception:
        is_avail = False

    return {
        "status": "ok",
        "service": "Sentinel Backend",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "message": "Sentinel backend is running successfully.",
        "llm": {
            "provider": display_name,
            "model": settings.LLM_MODEL,
            "api_key_required": False if prov_setting in ("local_ollama", "mock") else bool(settings.LLM_API_KEY),
            "provider_status": "Available" if is_avail else "Unavailable",
        },
    }


@router.get("/health/llm", tags=["Health"])
def get_llm_health():
    """
    Detailed non-blocking health check for local Ollama LLM service:
    Verifies reachability, configured model existence, and mini inference.
    """
    diag = check_local_llm_health()
    return {
        "status": "ok" if diag.get("available") else "degraded",
        "provider": "Local Ollama",
        "model": settings.LLM_MODEL,
        "api_key_required": False,
        "diagnostics": diag,
    }
