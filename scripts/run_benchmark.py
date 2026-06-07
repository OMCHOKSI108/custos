#!/usr/bin/env python3
"""
run_benchmark.py — Custos Cost/Latency Benchmark

Sends 120 queries (100 unique + 20 paraphrased duplicates) to the Custos /chat
endpoint and measures cost savings, cache hit rate, latency, and model distribution.

Usage:
    python scripts/run_benchmark.py --url http://localhost:8000
    python scripts/run_benchmark.py --url https://custos-lqtf.onrender.com --user-id bench
    python scripts/run_benchmark.py --dry-run

Prerequisites:
    pip install requests
    Clear caches before running:  curl -X DELETE <url>/cache
"""

import argparse
import json
import statistics
import sys
import time
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("ERROR: 'requests' library required. Install with: pip install requests")
    sys.exit(1)


# ─── Query Dataset ───────────────────────────────────────────────────────────
# 40 simple (factual lookups, definitions)
# 30 medium (technical explanations, how-tos)
# 30 complex (analytical, multi-part, code-heavy)

SIMPLE_QUERIES = [
    "What is Python?",
    "Define REST API",
    "Who created Linux?",
    "What is HTTP?",
    "Define machine learning",
    "What is a variable?",
    "Who invented the internet?",
    "What is JSON?",
    "Define an algorithm",
    "What is CSS?",
    "What is a database?",
    "Define cloud computing",
    "What is an array?",
    "Who created JavaScript?",
    "What is HTML?",
    "Define encryption",
    "What is a function?",
    "What is SQL?",
    "Define open source",
    "What is Git?",
    "What is a compiler?",
    "Define bandwidth",
    "What is Kubernetes?",
    "What is a boolean?",
    "Define latency",
    "What is TCP/IP?",
    "What is RAM?",
    "Define recursion",
    "What is an API key?",
    "What is Docker?",
    "What is a loop?",
    "Define DevOps",
    "What is HTTPS?",
    "What is a string?",
    "What is a pointer?",
    "Define SaaS",
    "What is a hash function?",
    "What is OAuth?",
    "Define a tuple",
    "What is YAML?",
]

MEDIUM_QUERIES = [
    "Explain how REST APIs work with examples",
    "How does garbage collection work in Java?",
    "Describe the differences between SQL and NoSQL databases",
    "How do you implement authentication in a web app?",
    "Explain how DNS resolution works step by step",
    "What are the main differences between TCP and UDP?",
    "How does HTTPS encryption protect data in transit?",
    "Explain how caching improves application performance",
    "Describe how a load balancer distributes traffic",
    "How do database indexes improve query performance?",
    "Explain how WebSockets differ from HTTP polling",
    "How does containerization work and why is Docker popular?",
    "Describe the CAP theorem and its implications for databases",
    "How do message queues work in distributed systems?",
    "Explain the concept of eventual consistency",
    "How does OAuth 2.0 authorization flow work?",
    "Describe how a reverse proxy works and when to use one",
    "How does rate limiting protect APIs from abuse?",
    "Explain how CI/CD pipelines automate software delivery",
    "How do content delivery networks reduce latency?",
    "Describe how connection pooling works in databases",
    "How does two-factor authentication improve security?",
    "Explain the publish-subscribe messaging pattern",
    "How does horizontal scaling differ from vertical scaling?",
    "Describe how feature flags enable gradual rollouts",
    "How do GraphQL queries differ from REST endpoints?",
    "Explain how gRPC uses protocol buffers for serialization",
    "How does a bloom filter work and where is it used?",
    "Describe how database sharding distributes data",
    "How do service meshes manage microservice communication?",
]

COMPLEX_QUERIES = [
    "Analyze the trade-offs between microservice and monolithic architectures for a high-traffic e-commerce platform",
    "Design a distributed caching strategy for a global application serving 10 million users with eventual consistency requirements",
    "Compare and contrast event sourcing vs CRUD for a financial transaction system, including auditability and performance implications",
    "Evaluate the pros and cons of using Kubernetes vs serverless for a startup with unpredictable traffic patterns and a small team",
    "Architect a real-time recommendation engine that processes user behavior events with sub-100ms latency requirements",
    "Analyze the security implications of implementing a zero-trust network architecture in a hybrid cloud environment",
    "Design a data pipeline for processing 1TB of daily log data with requirements for both real-time dashboards and batch analytics",
    "Compare the trade-offs between strong consistency and availability in a multi-region database deployment",
    "Evaluate different strategies for implementing backward-compatible API versioning in a platform with 500+ consumers",
    "Design a circuit breaker pattern implementation for a microservices system with cascading failure scenarios",
    "Analyze the performance implications of different serialization formats (JSON, Protocol Buffers, MessagePack) for high-throughput APIs",
    "Architect a multi-tenant SaaS platform with data isolation, custom domains, and per-tenant rate limiting",
    "Compare blue-green deployment vs canary releases vs rolling updates for a critical payment processing service",
    "Design an observability strategy combining distributed tracing, structured logging, and metrics for a 50-service architecture",
    "Evaluate the trade-offs between embedded databases (SQLite, RocksDB) vs client-server databases for edge computing scenarios",
    "Analyze how to implement a saga pattern for managing distributed transactions across five microservices with compensation logic",
    "Design a feature store architecture for a machine learning platform serving both batch training and real-time inference",
    "Compare approaches for implementing end-to-end encryption in a messaging app while maintaining server-side search capability",
    "Architect a system for handling 100K concurrent WebSocket connections with message ordering guarantees and exactly-once delivery",
    "Evaluate different approaches to implementing a custom query language for a domain-specific analytics platform",
    "Design a migration strategy for moving a 5-year-old monolith with 200 database tables to microservices without downtime",
    "Analyze the trade-offs of implementing CRDTs vs operational transforms for a collaborative document editing system",
    "Compare strategies for implementing multi-region active-active replication with conflict resolution in PostgreSQL",
    "Design a cost-optimized architecture for an LLM-powered application that handles 1M requests/day across multiple model providers",
    "Evaluate the security and performance implications of different container runtime options (containerd, gVisor, Kata Containers)",
    "Architect a plugin system for a developer tools platform that supports hot-reloading, sandboxed execution, and versioned APIs",
    "Analyze how to implement a distributed rate limiter that works across 10 server instances with minimal Redis dependencies",
    "Design a testing strategy for a system with 20 microservices including contract testing, chaos engineering, and performance testing",
    "Compare the trade-offs between pull-based (Prometheus) and push-based (StatsD/Graphite) monitoring for cloud-native applications",
    "Evaluate approaches for implementing a custom scheduler for batch ML training jobs with GPU affinity and preemption support",
]

# 20 paraphrased duplicates to test semantic caching
# Each maps to an original query above
PARAPHRASED_DUPLICATES = [
    # Simple paraphrases
    ("simple", "Can you explain what Python is?"),                          # → "What is Python?"
    ("simple", "Give me the definition of a REST API"),                     # → "Define REST API"
    ("simple", "Tell me who created the Linux operating system"),           # → "Who created Linux?"
    ("simple", "What exactly is JSON format?"),                             # → "What is JSON?"
    ("simple", "Could you define what machine learning means?"),            # → "Define machine learning"
    ("simple", "Explain what a database is"),                               # → "What is a database?"
    ("simple", "What does the term cloud computing mean?"),                 # → "Define cloud computing"
    ("simple", "Tell me what Git is used for"),                             # → "What is Git?"
    ("simple", "Can you explain what Docker does?"),                        # → "What is Docker?"
    ("simple", "What is the definition of encryption?"),                    # → "Define encryption"
    ("simple", "Describe what an algorithm is"),                            # → "Define an algorithm"
    ("simple", "What does HTML stand for and what is it?"),                 # → "What is HTML?"
    # Medium paraphrases
    ("medium", "Can you walk me through how REST APIs function?"),          # → "Explain how REST APIs work..."
    ("medium", "What are the key differences between SQL and NoSQL?"),      # → "Describe the differences..."
    ("medium", "How does DNS work when you type a URL?"),                   # → "Explain how DNS resolution..."
    ("medium", "Explain the difference between TCP and UDP protocols"),     # → "What are the main differences..."
    ("medium", "How does caching help make applications faster?"),          # → "Explain how caching improves..."
    # Complex paraphrases
    ("complex", "What are the advantages and disadvantages of microservices vs monoliths for a large e-commerce site?"),
    ("complex", "How would you design a caching system for a globally distributed app with millions of users?"),
    ("complex", "Compare event sourcing and CRUD architectures for handling financial transactions"),
]


# ─── Benchmark Runner ────────────────────────────────────────────────────────

def send_query(
    url: str, query: str, user_id: str, timeout: int = 120
) -> Optional[Dict]:
    """Send a single query to the /chat endpoint. Returns response dict or None on error."""
    try:
        resp = requests.post(
            f"{url}/chat",
            json={"query": query, "user_id": user_id},
            timeout=timeout,
        )
        if resp.status_code == 429:
            # Rate limited — wait and retry once
            retry_after = resp.json().get("detail", {}).get("retry_after", 10)
            print(f"  ⏳ Rate limited, waiting {retry_after}s...")
            time.sleep(min(retry_after, 60))
            resp = requests.post(
                f"{url}/chat",
                json={"query": query, "user_id": user_id},
                timeout=timeout,
            )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request failed: {e}")
        return None


def run_benchmark(url: str, user_id: str, delay: float = 1.0) -> Dict:
    """Run the full benchmark and return collected metrics."""
    results = []

    # Build ordered query list: all unique queries, then paraphrased duplicates
    queries = []
    for q in SIMPLE_QUERIES:
        queries.append(("simple", q))
    for q in MEDIUM_QUERIES:
        queries.append(("medium", q))
    for q in COMPLEX_QUERIES:
        queries.append(("complex", q))
    for category, q in PARAPHRASED_DUPLICATES:
        queries.append((f"{category}_dup", q))

    total = len(queries)
    print(f"\n{'='*70}")
    print(f"  Custos Benchmark — {total} queries")
    print(f"  Target: {url}")
    print(f"  User:   {user_id}")
    print(f"{'='*70}\n")

    for i, (category, query) in enumerate(queries, 1):
        label = f"[{i:3d}/{total}] ({category:12s})"
        preview = query[:60] + ("..." if len(query) > 60 else "")
        print(f"{label} {preview}", end="", flush=True)

        start = time.time()
        resp = send_query(url, query, user_id)
        elapsed_ms = (time.time() - start) * 1000

        if resp is None:
            print(f" — FAILED")
            results.append({
                "category": category,
                "query": query,
                "success": False,
            })
        else:
            cost = resp.get("cost_usd", 0.0)
            model = resp.get("model", "unknown")
            cache_hit = resp.get("cache_hit", False)
            cache_type = resp.get("cache_type", "none")
            latency = resp.get("latency_ms", elapsed_ms)

            cache_label = f"cache:{cache_type}" if cache_hit else model.split("-")[-1]
            print(f" — ${cost:.6f} | {cache_label} | {latency:.0f}ms")

            results.append({
                "category": category,
                "query": query,
                "success": True,
                "cost": cost,
                "model": model,
                "cache_hit": cache_hit,
                "cache_type": cache_type,
                "latency_ms": latency,
                "client_latency_ms": elapsed_ms,
            })

        # Throttle to avoid rate limits
        time.sleep(delay)

    return {"results": results, "url": url, "user_id": user_id}


# ─── Results Formatting ─────────────────────────────────────────────────────

# Pricing for baseline estimate (all queries to gemini-2.5-pro)
PRO_INPUT_PER_1K = 0.00125
PRO_OUTPUT_PER_1K = 0.005
# Average tokens by category (estimated)
TOKEN_ESTIMATES = {
    "simple": {"input": 15, "output": 80},
    "medium": {"input": 35, "output": 150},
    "complex": {"input": 60, "output": 250},
}


def estimate_baseline_cost(category: str) -> float:
    """Estimate what this query would cost if sent directly to gemini-2.5-pro."""
    base = category.replace("_dup", "")
    tokens = TOKEN_ESTIMATES.get(base, TOKEN_ESTIMATES["medium"])
    return (tokens["input"] / 1000 * PRO_INPUT_PER_1K) + (
        tokens["output"] / 1000 * PRO_OUTPUT_PER_1K
    )


def format_results(data: Dict) -> str:
    """Generate a markdown-formatted summary table."""
    results = [r for r in data["results"] if r["success"]]
    failed = [r for r in data["results"] if not r["success"]]

    if not results:
        return "## Benchmark Failed\n\nNo successful requests."

    # Aggregate metrics
    total_cost = sum(r["cost"] for r in results)
    total_baseline = sum(estimate_baseline_cost(r["category"]) for r in results)
    costs = [r["cost"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    cache_hits = [r for r in results if r["cache_hit"]]
    exact_hits = [r for r in cache_hits if r["cache_type"] == "exact"]
    semantic_hits = [r for r in cache_hits if r["cache_type"] == "semantic"]

    models_used = {}
    for r in results:
        m = r["model"]
        models_used[m] = models_used.get(m, 0) + 1

    # Per-category breakdown
    categories = {}
    for r in results:
        base_cat = r["category"].replace("_dup", "")
        if base_cat not in categories:
            categories[base_cat] = {"cost": 0, "baseline": 0, "count": 0, "cached": 0}
        categories[base_cat]["cost"] += r["cost"]
        categories[base_cat]["baseline"] += estimate_baseline_cost(r["category"])
        categories[base_cat]["count"] += 1
        if r["cache_hit"]:
            categories[base_cat]["cached"] += 1

    # Format output
    lines = []
    lines.append("")
    lines.append(f"# Custos Benchmark Results")
    lines.append("")
    lines.append(f"**Target:** {data['url']}")
    lines.append(f"**Queries:** {len(results)} successful, {len(failed)} failed")
    lines.append("")

    # Cost summary
    savings_pct = (1 - total_cost / total_baseline) * 100 if total_baseline > 0 else 0
    lines.append("## Cost Summary")
    lines.append("")
    lines.append("| Metric | Baseline (all Pro) | Custos | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Total cost | ${total_baseline:.4f} | ${total_cost:.4f} "
        f"| **−{savings_pct:.1f}%** |"
    )
    lines.append(
        f"| Avg cost/request | ${total_baseline/len(results):.6f} "
        f"| ${total_cost/len(results):.6f} | |"
    )
    lines.append(
        f"| Cache hit rate | 0% | {len(cache_hits)/len(results)*100:.1f}% "
        f"({len(cache_hits)}/{len(results)}) | |"
    )
    lines.append("")

    # Latency
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95_idx = int(len(sorted_lat) * 0.95)
    p95 = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]

    lines.append("## Latency")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Avg | {statistics.mean(latencies):.0f} ms |")
    lines.append(f"| P50 | {p50:.0f} ms |")
    lines.append(f"| P95 | {p95:.0f} ms |")
    if cache_hits:
        cache_lats = [r["latency_ms"] for r in cache_hits]
        lines.append(f"| Cache hit avg | {statistics.mean(cache_lats):.1f} ms |")
    lines.append("")

    # Model distribution
    lines.append("## Model Distribution")
    lines.append("")
    lines.append("| Model | Requests | % |")
    lines.append("|---|---|---|")
    for model, count in sorted(models_used.items(), key=lambda x: -x[1]):
        lines.append(f"| `{model}` | {count} | {count/len(results)*100:.1f}% |")
    lines.append(f"| Cached (zero cost) | {len(cache_hits)} | {len(cache_hits)/len(results)*100:.1f}% |")
    lines.append("")

    # Cache breakdown
    lines.append("## Cache Breakdown")
    lines.append("")
    lines.append("| Layer | Hits |")
    lines.append("|---|---|")
    lines.append(f"| Exact match | {len(exact_hits)} |")
    lines.append(f"| Semantic match | {len(semantic_hits)} |")
    lines.append(f"| **Total** | **{len(cache_hits)}** |")
    lines.append("")

    # Per-category
    lines.append("## Cost by Category")
    lines.append("")
    lines.append("| Category | Requests | Baseline | Custos | Savings | Cached |")
    lines.append("|---|---|---|---|---|---|")
    for cat in ["simple", "medium", "complex"]:
        if cat in categories:
            c = categories[cat]
            sav = (1 - c["cost"] / c["baseline"]) * 100 if c["baseline"] > 0 else 0
            lines.append(
                f"| {cat.capitalize()} | {c['count']} | ${c['baseline']:.4f} "
                f"| ${c['cost']:.4f} | {sav:.1f}% | {c['cached']} |"
            )
    lines.append("")

    if failed:
        lines.append(f"## Failures")
        lines.append(f"")
        lines.append(f"{len(failed)} queries failed to get a response.")
        lines.append("")

    return "\n".join(lines)


def dry_run():
    """Print all queries without sending them."""
    queries = []
    for q in SIMPLE_QUERIES:
        queries.append(("simple", q))
    for q in MEDIUM_QUERIES:
        queries.append(("medium", q))
    for q in COMPLEX_QUERIES:
        queries.append(("complex", q))
    for category, q in PARAPHRASED_DUPLICATES:
        queries.append((f"{category}_dup", q))

    print(f"\n{'='*70}")
    print(f"  Custos Benchmark — DRY RUN ({len(queries)} queries)")
    print(f"{'='*70}\n")

    for i, (category, query) in enumerate(queries, 1):
        preview = query[:80] + ("..." if len(query) > 80 else "")
        print(f"  [{i:3d}] ({category:12s}) {preview}")

    print(f"\n  Total: {len(SIMPLE_QUERIES)} simple + {len(MEDIUM_QUERIES)} medium "
          f"+ {len(COMPLEX_QUERIES)} complex + {len(PARAPHRASED_DUPLICATES)} duplicates "
          f"= {len(queries)} queries")
    print(f"  Use without --dry-run to execute.\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Custos benchmark — measure cost savings from intelligent LLM routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py --url http://localhost:8000
  python scripts/run_benchmark.py --url https://custos-lqtf.onrender.com --user-id bench
  python scripts/run_benchmark.py --dry-run

Clear caches before running for clean results:
  curl -X DELETE http://localhost:8000/cache
        """,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Custos server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--user-id",
        default="benchmark",
        help="User ID for requests (default: benchmark)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print queries without sending them",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    # Verify server is reachable
    try:
        health = requests.get(f"{args.url}/health", timeout=10)
        health.raise_for_status()
        print(f"✓ Server reachable: {args.url}")
        info = health.json()
        if info.get("mock_mode"):
            print("  ⚠ MOCK_MODE is enabled — costs will be simulated, not real")
    except Exception as e:
        print(f"✗ Cannot reach server at {args.url}: {e}")
        sys.exit(1)

    data = run_benchmark(args.url, args.user_id, delay=args.delay)
    summary = format_results(data)

    print(f"\n{'='*70}")
    print(summary)
    print(f"{'='*70}\n")

    # Also save to file
    output_file = "benchmark_results.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
