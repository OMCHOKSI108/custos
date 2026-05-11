"""
router.py - Main LLM Router

Supports multiple LLM providers (Gemini, Groq) with dynamic model discovery.
Routes simple queries to cheap models, complex to expensive ones.
"""

import time
import random
import asyncio
from typing import Dict, Any, Optional

from app.analyzer import compute_complexity, select_model
from app.cache import ExactCache
from app.semantic_cache import SemanticCache
from app.budget import BudgetEnforcer, BudgetExceededError
from app.rate_limiter import RateLimiter
from app.logger import RequestLogger
from app.config import (
    GEMINI_API_KEY,
    MOCK_MODE,
    MODEL_COSTS,
    GEMINI_FLASH,
    GEMINI_PRO,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    LLM_PROVIDER,
    GROQ_API_KEY,
)
from app.model_discovery import discover_models
import app.groq_client as groq

_gemini_client = None
GEMINI_AVAILABLE = False

if not MOCK_MODE and GEMINI_API_KEY:
    try:
        from google import genai as _genai_module

        _gemini_client = _genai_module.Client(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        flash_model, pro_model = discover_models(_gemini_client)
        GEMINI_FLASH_DISCOVERED = flash_model
        GEMINI_PRO_DISCOVERED = pro_model
        print(
            "Gemini client ready | flash={}, pro={}".format(
                GEMINI_FLASH_DISCOVERED, GEMINI_PRO_DISCOVERED
            )
        )
    except Exception as e:
        print("Gemini init failed: {}".format(e))
        GEMINI_FLASH_DISCOVERED = GEMINI_FLASH
        GEMINI_PRO_DISCOVERED = GEMINI_PRO
else:
    GEMINI_FLASH_DISCOVERED = GEMINI_FLASH
    GEMINI_PRO_DISCOVERED = GEMINI_PRO

groq.init_groq(GROQ_API_KEY)

MOCK_RESPONSES = [
    "This is a mock response. Set MOCK_MODE=false and add GEMINI_API_KEY to get real answers.",
    "Mock mode is active. The routing, caching, and cost tracking are all real - only the LLM call is fake.",
    "To get real responses: set MOCK_MODE=false and add your API key, then restart.",
]


def _calculate_cost(
    model: str, tokens_in: int, tokens_out: int, provider: str = "gemini"
) -> float:
    if provider == "groq":
        return groq.calculate_cost(model, tokens_in, tokens_out)
    pricing = MODEL_COSTS.get(model, MODEL_COSTS[GEMINI_FLASH])
    return (tokens_in / 1000 * pricing["input"]) + (
        tokens_out / 1000 * pricing["output"]
    )


class LLMRouter:
    def __init__(self):
        self.exact_cache = ExactCache()
        self.semantic_cache = SemanticCache()
        self.budget = BudgetEnforcer()
        self.rate_limiter = RateLimiter()
        self.logger = RequestLogger()
        self._ml_router = None

    def load_ml_router(self, ml_router):
        self._ml_router = ml_router

    def _call_gemini_with_retry(self, query: str, model: str) -> Dict:
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                response = _gemini_client.models.generate_content(
                    model=model,
                    contents=query,
                )
                text = response.text
                usage = getattr(response, "usage_metadata", None)
                tokens_in = getattr(usage, "prompt_token_count", 50) if usage else 50
                tokens_out = (
                    getattr(usage, "candidates_token_count", 80) if usage else 80
                )
                return {
                    "response": text,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }

            except Exception as e:
                last_exception = e
                err_str = str(e)

                if (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "quota" in err_str.lower()
                ):
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2**attempt)
                        print(
                            "Gemini 429 on attempt {}/{} - waiting {}s before retry".format(
                                attempt + 1, MAX_RETRIES, delay
                            )
                        )
                        time.sleep(delay)
                        continue

                break

        raise last_exception

    def _mock_call(self, query: str, model: str) -> Dict:
        time.sleep(random.uniform(0.05, 0.12))
        tokens_in = max(10, len(query.split()) + 5)
        tokens_out = random.randint(60, 160)
        return {
            "response": random.choice(MOCK_RESPONSES),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    def process(
        self,
        query: str,
        user_id: str = "anonymous",
        force_model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        active_provider = (provider or LLM_PROVIDER).lower()

        rate_check = self.rate_limiter.check(user_id)
        if not rate_check["allowed"]:
            raise ValueError(
                "Rate limit exceeded. Limit: {}/hour. Retry after {}s.".format(
                    rate_check["limit"], rate_check["retry_after"]
                )
            )

        self.budget.check_budget()

        complexity_result = compute_complexity(query)
        complexity_score = complexity_result["composite"]

        if force_model:
            model = force_model
        elif self._ml_router and self._ml_router.is_trained:
            ml_result = self._ml_router.predict(query)
            model = (
                ml_result["model"]
                if ml_result["confidence"] > 0.7
                else select_model(complexity_score)
            )
        else:
            model = select_model(complexity_score)

        cached = self.exact_cache.get(query, model)
        if cached:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.log(
                user_id,
                query,
                model,
                complexity_score,
                0,
                0,
                0.0,
                True,
                "exact",
                latency_ms,
            )
            return {
                "response": cached["response"],
                "model": model,
                "cost_usd": 0.0,
                "cache_hit": True,
                "cache_type": "exact",
                "complexity_score": complexity_score,
                "tokens": {"input": 0, "output": 0},
                "latency_ms": round(latency_ms, 2),
                "provider": active_provider,
            }

        sem_cached = self.semantic_cache.get(query)
        if sem_cached:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.log(
                user_id,
                query,
                sem_cached["model"],
                complexity_score,
                0,
                0,
                0.0,
                True,
                "semantic",
                latency_ms,
            )
            return {
                "response": sem_cached["response"],
                "model": sem_cached["model"],
                "cost_usd": 0.0,
                "cache_hit": True,
                "cache_type": "semantic",
                "similarity_score": sem_cached.get("similarity_score"),
                "complexity_score": complexity_score,
                "tokens": {"input": 0, "output": 0},
                "latency_ms": round(latency_ms, 2),
                "provider": sem_cached.get("provider", active_provider),
            }

        api_error = None
        try:
            if MOCK_MODE:
                llm_result = self._mock_call(query, model)
            elif active_provider == "groq":
                if not groq.GROQ_AVAILABLE:
                    raise ValueError("Groq not configured. Set GROQ_API_KEY.")
                llm_result = groq._call_groq_with_retry(query, model)
            elif GEMINI_AVAILABLE:
                llm_result = self._call_gemini_with_retry(query, model)
            else:
                llm_result = self._mock_call(query, model)

        except Exception as e:
            api_error = str(e)
            err_str = str(e)
            latency_ms = (time.time() - start_time) * 1000

            if active_provider == "groq":
                user_msg = "Groq API error: {}".format(err_str[:120])
            elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                user_msg = (
                    "Gemini free tier quota exceeded. "
                    "Wait a minute and retry, or add billing for higher limits."
                )
            elif "API_KEY" in err_str or "401" in err_str:
                user_msg = "Invalid Gemini API key. Check GEMINI_API_KEY."
            elif "404" in err_str:
                user_msg = (
                    "Model not found: {}. Check config for correct model names.".format(
                        model
                    )
                )
            else:
                user_msg = "API error: {}".format(err_str[:120])

            return {
                "response": user_msg,
                "model": model,
                "cost_usd": 0.0,
                "cache_hit": False,
                "cache_type": "none",
                "complexity_score": complexity_score,
                "tokens": {"input": 0, "output": 0},
                "latency_ms": round(latency_ms, 2),
                "error": True,
                "error_type": "api_error",
                "provider": active_provider,
            }

        cost = _calculate_cost(
            model, llm_result["tokens_in"], llm_result["tokens_out"], active_provider
        )

        self.budget.record_cost(cost)
        self.exact_cache.set(query, model, llm_result["response"], cost)
        self.semantic_cache.set(query, llm_result["response"], model, cost)

        latency_ms = (time.time() - start_time) * 1000
        self.logger.log(
            user_id,
            query,
            model,
            complexity_score,
            llm_result["tokens_in"],
            llm_result["tokens_out"],
            cost,
            False,
            "none",
            latency_ms,
        )

        return {
            "response": llm_result["response"],
            "model": model,
            "cost_usd": round(cost, 8),
            "cache_hit": False,
            "cache_type": "none",
            "complexity_score": complexity_score,
            "complexity_factors": complexity_result["factors"],
            "tokens": {
                "input": llm_result["tokens_in"],
                "output": llm_result["tokens_out"],
            },
            "latency_ms": round(latency_ms, 2),
            "provider": active_provider,
        }

    def get_full_stats(self) -> Dict:
        from app.analytics import compute_analytics
        from app.config import LOG_FILE

        return {
            "exact_cache": self.exact_cache.stats(),
            "semantic_cache": self.semantic_cache.stats(),
            "budget": self.budget.status(),
            "rate_limiter": self.rate_limiter.stats(),
            "analytics": compute_analytics(LOG_FILE),
            "ml_router": {
                "trained": self._ml_router.is_trained if self._ml_router else False,
                "model_type": "RandomForest"
                if (self._ml_router and self._ml_router.is_trained)
                else "heuristic",
            },
            "gemini_available": GEMINI_AVAILABLE,
            "groq_available": groq.GROQ_AVAILABLE,
            "active_provider": LLM_PROVIDER,
            "mock_mode": MOCK_MODE,
            "discovered_models": {
                "flash": GEMINI_FLASH_DISCOVERED,
                "pro": GEMINI_PRO_DISCOVERED,
            },
        }
