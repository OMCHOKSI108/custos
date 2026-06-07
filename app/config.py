"""
config.py

IMPORTANT MODEL UPDATE (May 2026):
  gemini-2.0-flash  -> shutting down June 1 2026. Replaced by gemini-2.5-flash.
  gemini-2.5-flash  -> best free tier limits (15 RPM, ~500 RPD on free tier)
  gemini-2.5-pro    -> for complex queries (5 RPM free tier, use sparingly)

FREE TIER LIMITS (per project, as of 2025-2026):
  gemini-2.5-flash: 15 RPM, ~500 RPD
  gemini-2.5-pro:   5 RPM,  50 RPD

DYNAMIC MODEL DISCOVERY:
  At startup, Custos queries the Gemini API to find the latest available
  flash and pro models (e.g. gemini-3.0-flash). Falls back to these defaults.

PROVIDER SUPPORT:
  LLM_PROVIDER=gemini (default) or groq
  Groq requires GROQ_API_KEY from https://console.groq.com/keys
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_PRO = "gemini-2.5-pro"

MODEL_COSTS = {
    "gemini-2.5-flash": {"input": 0.0003, "output": 0.0024},
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
}

SIMPLE_THRESHOLD = 0.35
COMPLEX_THRESHOLD = 0.65

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.85"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "500"))

RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))

DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "10.0"))
HOURLY_BUDGET_USD = float(os.getenv("HOURLY_BUDGET_USD", "2.0"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "logs/requests.csv")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "2.0"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

CUSTOS_API_KEY = os.getenv("CUSTOS_API_KEY", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
