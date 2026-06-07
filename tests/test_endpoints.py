"""
tests/test_endpoints.py - Comprehensive system validation as pytest tests.

Covers every module: analyzer, budget, rate limiter, cache, full router pipeline, logger, stats.
Run with: python -m pytest tests/test_endpoints.py -v
"""

import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MOCK_MODE"] = "true"
os.environ["LLM_PROVIDER"] = "gemini"

import pytest
from app.router import LLMRouter
from app.budget import BudgetEnforcer, BudgetExceededError
from app.rate_limiter import RateLimiter
from app.analyzer import compute_complexity, select_model
from app.cache import ExactCache
from app.logger import RequestLogger
import tempfile


class TestAnalyzer:
    def test_simple_query_low_score(self):
        result = compute_complexity("What is 2+2?")
        assert result["composite"] < 0.30

    def test_complex_query_high_score(self):
        result = compute_complexity(
            "Analyze the architectural trade-offs between microservices and monolithic "
            "systems in distributed ML inference. Compare latency, scalability, complexity."
        )
        assert result["composite"] > 0.35

    def test_complex_beats_simple(self):
        simple = compute_complexity("What is 2+2?")
        complex = compute_complexity(
            "Analyze and evaluate the philosophical implications of artificial "
            "consciousness with a detailed comparison framework."
        )
        assert complex["composite"] > simple["composite"]

    def test_all_six_factors_present(self):
        result = compute_complexity("What is machine learning?")
        for key in ["length", "keywords", "questions", "code", "technical", "sentence"]:
            assert key in result["factors"]

    def test_score_in_range(self):
        for q in ["Hi", "What is AI?", "a" * 500]:
            result = compute_complexity(q)
            assert 0.0 <= result["composite"] <= 1.0

    def test_simple_routes_to_flash(self):
        assert "flash" in select_model(0.10).lower()

    def test_complex_routes_to_pro(self):
        assert "pro" in select_model(0.90).lower()


class TestBudgetEnforcer:
    def test_no_block_under_limit(self):
        b = BudgetEnforcer(daily_limit=5.0, hourly_limit=1.0)
        b.check_budget()

    def test_daily_block_when_exceeded(self):
        b = BudgetEnforcer(daily_limit=0.001, hourly_limit=100.0)
        b.record_cost(0.002)
        with pytest.raises(BudgetExceededError) as exc:
            b.check_budget()
        assert "Daily" in exc.value.message

    def test_hourly_block_when_exceeded(self):
        b = BudgetEnforcer(daily_limit=100.0, hourly_limit=0.001)
        b.record_cost(0.002)
        with pytest.raises(BudgetExceededError) as exc:
            b.check_budget()
        assert "Hourly" in exc.value.message

    def test_cost_accumulates(self):
        b = BudgetEnforcer(daily_limit=100.0, hourly_limit=100.0)
        b.record_cost(0.01)
        b.record_cost(0.02)
        b.record_cost(0.03)
        assert abs(b.status()["total_cost_usd"] - 0.06) < 1e-9

    def test_status_has_all_fields(self):
        b = BudgetEnforcer(daily_limit=5.0, hourly_limit=1.0)
        b.record_cost(0.5)
        s = b.status()
        for key in [
            "daily_spent_usd",
            "daily_limit_usd",
            "daily_remaining_usd",
            "hourly_spent_usd",
            "total_cost_usd",
            "total_requests",
        ]:
            assert key in s

    def test_update_limits_at_runtime(self):
        b = BudgetEnforcer(daily_limit=10.0, hourly_limit=2.0)
        b.update_limits(daily=0.001)
        b.record_cost(0.002)
        with pytest.raises(BudgetExceededError):
            b.check_budget()


class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(requests_per_hour=3)
        for _ in range(3):
            assert rl.check("user")["allowed"]

    def test_blocks_at_limit(self):
        rl = RateLimiter(requests_per_hour=3)
        for _ in range(3):
            rl.check("user")
        result = rl.check("user")
        assert not result["allowed"]
        assert result["reason"] == "hourly_rate_limit"

    def test_different_users_independent(self):
        rl = RateLimiter(requests_per_hour=1)
        rl.check("user1")
        rl.check("user1")
        assert rl.check("user2")["allowed"]

    def test_retry_after_on_block(self):
        rl = RateLimiter(requests_per_hour=1)
        rl.check("user")
        result = rl.check("user")
        assert result.get("retry_after", 0) > 0

    def test_stats_tracking(self):
        rl = RateLimiter(requests_per_hour=2)
        rl.check("u")
        rl.check("u")
        rl.check("u")
        stats = rl.stats()
        assert stats["total_blocked"] >= 1
        assert stats["total_allowed"] == 2


class TestExactCache:
    def test_miss_on_empty(self):
        c = ExactCache()
        assert c.get("hello", "flash") is None

    def test_set_then_hit(self):
        c = ExactCache()
        c.set("q", "flash", "response", 0.001)
        assert c.get("q", "flash") is not None

    def test_different_model_is_miss(self):
        c = ExactCache()
        c.set("q", "flash", "response", 0.001)
        assert c.get("q", "pro") is None

    def test_ttl_expiry(self):
        c = ExactCache(ttl=0)
        c.set("q", "flash", "r", 0.0)
        time.sleep(0.01)
        assert c.get("q", "flash") is None

    def test_lru_eviction(self):
        c = ExactCache(max_size=3)
        c.set("q1", "flash", "r1", 0.0)
        c.set("q2", "flash", "r2", 0.0)
        c.set("q3", "flash", "r3", 0.0)
        c.set("q4", "flash", "r4", 0.0)
        assert c.get("q1", "flash") is None
        assert c.get("q4", "flash") is not None

    def test_clear(self):
        c = ExactCache()
        c.set("q", "flash", "r", 0.0)
        c.clear()
        assert c.get("q", "flash") is None

    def test_hit_rate_tracking(self):
        c = ExactCache()
        c.set("q1", "flash", "r1", 0.0)
        c.get("q1", "flash")
        c.get("q2", "flash")
        assert c.stats()["hit_rate_pct"] == 50.0


class TestLogger:
    def test_log_and_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            tmpfile = f.name
        try:
            log = RequestLogger(log_file=tmpfile)
            log.log(
                "u1",
                "What is AI?",
                "gemini-1.5-flash",
                0.25,
                20,
                80,
                0.00012,
                False,
                "none",
                145.3,
            )
            log.log(
                "u1", "Same", "gemini-1.5-flash", 0.25, 0, 0, 0.0, True, "exact", 3.1
            )
            log.log(
                "u2",
                "Different",
                "gemini-1.5-pro",
                0.80,
                50,
                200,
                0.003,
                False,
                "none",
                890.0,
            )
            rows = log.read_all()
            assert len(rows) == 3
            assert rows[0]["user_id"] == "u2"
            csv_str = log.export_csv_string()
            assert "timestamp" in csv_str
        finally:
            os.unlink(tmpfile)


class TestRouterPipeline:
    def setup_method(self):
        self.router = LLMRouter()

    def test_basic_response(self):
        result = self.router.process("What is Python?", user_id="test")
        assert "response" in result
        assert len(result["response"]) > 0

    def test_metadata_fields(self):
        result = self.router.process("Explain ML", user_id="test")
        for field in [
            "response",
            "model",
            "cost_usd",
            "cache_hit",
            "cache_type",
            "complexity_score",
            "tokens",
            "latency_ms",
        ]:
            assert field in result

    def test_simple_query_uses_flash(self):
        result = self.router.process("What is 2+2?", user_id="test")
        assert "flash" in result["model"].lower()

    def test_repeat_query_is_cache_hit(self):
        q = "What is Python?"
        self.router.process(q, user_id="test")
        r2 = self.router.process(q, user_id="test")
        assert r2["cache_hit"] is True
        assert r2["cache_type"] == "exact"
        assert r2["cost_usd"] == 0.0

    def test_budget_block(self):
        router2 = LLMRouter()
        router2.budget.update_limits(daily=0.000001)
        router2.budget.record_cost(0.001)
        with pytest.raises((BudgetExceededError, ValueError)):
            router2.process("any query", user_id="test")

    def test_rate_limit_block(self):
        router3 = LLMRouter()
        router3.rate_limiter = RateLimiter(requests_per_hour=1)
        router3.process("first", user_id="limited")
        with pytest.raises(ValueError, match="Rate limit"):
            router3.process("second", user_id="limited")

    def test_complexity_score_in_range(self):
        for q in ["Hi", "What is AI?", "Analyze trade-offs in depth"]:
            result = self.router.process(q, user_id="test")
            assert 0.0 <= result["complexity_score"] <= 1.0

    def test_full_stats_sections(self):
        self.router.process("test query", user_id="test")
        stats = self.router.get_full_stats()
        for section in ["exact_cache", "semantic_cache", "budget", "rate_limiter"]:
            assert section in stats

    def test_response_includes_provider(self):
        result = self.router.process("hello", user_id="test")
        assert "provider" in result
        assert result["provider"] == "gemini"

    def test_groq_provider_not_available_in_mock(self):
        import app.groq_client as groq

        assert groq.GROQ_AVAILABLE is False
