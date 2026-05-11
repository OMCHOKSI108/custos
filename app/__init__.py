"""
Custos - Intelligent LLM Gateway

Usage as a package:
    from app import Custos

    client = Custos(
        gemini_api_key="...",
        provider="gemini",
        daily_budget=10.0,
    )

    result = client.query("What is Python?", user_id="user_1")
    print(result["response"])
"""

from typing import Optional, Dict, Any


class Custos:
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        provider: str = "gemini",
        daily_budget: float = 10.0,
        hourly_budget: float = 2.0,
        rate_limit: int = 100,
    ):
        import os
        from app.router import LLMRouter
        from app.ml_router import MLRouter
        from app.budget import BudgetEnforcer
        from app.rate_limiter import RateLimiter

        if gemini_api_key:
            os.environ["GEMINI_API_KEY"] = gemini_api_key
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ.setdefault("MOCK_MODE", "true" if not gemini_api_key else "false")
        os.environ["LLM_PROVIDER"] = provider
        os.environ["DAILY_BUDGET_USD"] = str(daily_budget)
        os.environ["HOURLY_BUDGET_USD"] = str(hourly_budget)
        os.environ["RATE_LIMIT_PER_HOUR"] = str(rate_limit)

        self._ml_router = MLRouter()
        self._router = LLMRouter()
        self._router.load_ml_router(self._ml_router)

    def query(
        self,
        query: str,
        user_id: str = "default",
        force_model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._router.process(
            query=query,
            user_id=user_id,
            force_model=force_model,
            provider=provider,
        )

    def stats(self) -> Dict[str, Any]:
        return self._router.get_full_stats()

    def clear_cache(self):
        self._router.exact_cache.clear()
        self._router.semantic_cache.clear()
