"""
groq_client.py - Groq API Provider

Wraps the Groq SDK for use as an alternative LLM provider.
Users can switch between Gemini and Groq via config or request parameter.

Groq offers fast inference on open-source models:
  - mixtral-8x7b-32768 (default)
  - llama-3.3-70b-versatile
  - llama-3.1-8b-instant
  - gemma2-9b-it

Get an API key: https://console.groq.com/keys
"""

import time
from typing import Dict, Optional

GROQ_MODEL_CHEAP = "mixtral-8x7b-32768"
GROQ_MODEL_EXPENSIVE = "llama-3.3-70b-versatile"

# Cost per 1M tokens (USD) for common Groq models
GROQ_MODEL_COSTS = {
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}

_groq_client = None
GROQ_AVAILABLE = False
GROQ_API_KEY = ""
GROQ_MAX_RETRIES = 3
GROQ_RETRY_BASE_DELAY = 2.0


def init_groq(api_key: str):
    global _groq_client, GROQ_AVAILABLE, GROQ_API_KEY
    GROQ_API_KEY = api_key
    if not api_key:
        GROQ_AVAILABLE = False
        return
    try:
        from groq import Groq

        _groq_client = Groq(api_key=api_key)
        GROQ_AVAILABLE = True
        print("Groq client ready | model={}".format(GROQ_MODEL_CHEAP))
    except Exception as e:
        GROQ_AVAILABLE = False
        print("Groq init failed: {}".format(e))


def _call_groq_with_retry(query: str, model: str) -> Dict:
    last_exception = None
    for attempt in range(GROQ_MAX_RETRIES):
        try:
            response = _groq_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": query}],
            )
            text = response.choices[0].message.content
            usage = getattr(response, "usage", None)
            tokens_in = getattr(usage, "prompt_tokens", 50) if usage else 50
            tokens_out = getattr(usage, "completion_tokens", 80) if usage else 80
            return {"response": text, "tokens_in": tokens_in, "tokens_out": tokens_out}

        except Exception as e:
            last_exception = e
            err_str = str(e)
            if (
                "429" in err_str
                or "rate" in err_str.lower()
                or "quota" in err_str.lower()
            ):
                if attempt < GROQ_MAX_RETRIES - 1:
                    delay = GROQ_RETRY_BASE_DELAY * (2**attempt)
                    print(
                        "Groq 429 on attempt {}/{} - waiting {}s before retry".format(
                            attempt + 1, GROQ_MAX_RETRIES, delay
                        )
                    )
                    time.sleep(delay)
                    continue
            break

    raise last_exception


def get_model_costs(model: str) -> Dict:
    return GROQ_MODEL_COSTS.get(model, {"input": 0.24, "output": 0.24})


def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = get_model_costs(model)
    return (tokens_in / 1000000 * pricing["input"]) + (
        tokens_out / 1000000 * pricing["output"]
    )
