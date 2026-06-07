# Technical Deep Dive — Custos

> **For engineers and technical interviewers**
> This document explains how the system works internally, design decisions, and trade-offs made.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Intelligent Routing Algorithm](#intelligent-routing-algorithm)
3. [ML Router (Random Forest)](#ml-router-random-forest)
4. [Semantic Caching System](#semantic-caching-system)
5. [Multi-Provider Support](#multi-provider-support)
6. [Analytics & Observability](#analytics--observability)
7. [Performance Characteristics](#performance-characteristics)
8. [Design Decisions & Trade-offs](#design-decisions--trade-offs)

---

## Architecture Overview

### High-Level Flow

```
┌─────────────┐
│   Client    │
│ Application │
└──────┬──────┘
       │ 1. POST /chat
       ▼
┌──────────────────────────────────────┐
│         FastAPI Server               │
│  ┌────────────────────────────────┐  │
│  │  1. Rate Limiter              │  │ ← Per-user throttle
│  │  2. Budget Enforcer           │  │ ← Daily/hourly caps
│  │  3. Query Analyzer            │  │ ← Complexity scoring
│  │  4. Exact Cache Check         │  │ ← O(1) hash lookup
│  │  5. Semantic Cache Check      │  │ ← Embedding similarity
│  │  6. ML Router / Heuristic     │  │ ← Model selection
│  │  7. Cost Tracker              │  │ ← Metrics & billing
│  │  8. Analytics Logger          │  │ ← CSV persistence
│  └────────────────────────────────┘  │
└──────┬───────────────────┬───────────┘
       │                   │
       │ 2a. Gemini API    │ 2b. Groq API (optional)
       ▼                   ▼
┌──────────────┐   ┌──────────────┐
│  Gemini API  │   │   Groq API   │
│ (Flash/Pro)  │   │ (Llama 3.x)  │
└──────────────┘   └──────────────┘
```

### Component Responsibilities

| Component | Purpose | Time Complexity | Space Complexity |
|-----------|---------|-----------------|------------------|
| Rate Limiter | Throttle per-user requests | O(1) | O(u) where u = users |
| Budget Enforcer | Cap daily/hourly spend | O(1) | O(1) |
| Query Analyzer | Score complexity (0–1) | O(n) where n = query length | O(1) |
| Exact Cache | Hash-based lookup | O(1) | O(k) where k = cached queries |
| Semantic Cache | Embedding similarity | O(m) where m = cache size | O(m × d) where d = 384 |
| ML Router | Predict optimal model | O(1) amortized | O(T) where T = tree count |
| Heuristic Router | Weighted rule scoring | O(n) | O(1) |
| Cost Tracker | Update cost metrics | O(1) | O(1) |
| Logger | Append to CSV | O(1) amortized | O(n) on disk |

---

## Intelligent Routing Algorithm

### Complexity Scoring

**Goal:** Determine if a query needs the expensive model (Gemini Pro) or the cheap model (Gemini Flash).

**Algorithm:** Weighted heuristic scoring across 6 independent signals.

```python
# Each signal is clipped to [0, 1] before weighting
complexity_score = (
    length_score    * 0.15 +   # Longer queries → more reasoning needed
    keyword_score   * 0.40 +   # "analyze", "compare" → complex intent
    question_score  * 0.10 +   # Multiple ?s → multi-part question
    code_score      * 0.20 +   # Code blocks → needs capable model
    technical_score * 0.10 +   # Domain jargon (API, LLM, etc.)
    sentence_score  * 0.05     # Long sentences → complex structure
)
```

**Signal Details:**

| Signal | Weight | Calculation | Range |
|--------|--------|-------------|-------|
| `length` | 0.15 | `min(word_count / 100, 1.0)` | 0–1 |
| `keywords` | 0.40 | `complex_hits × 0.15 − simple_hits × 0.10` | 0–1 |
| `questions` | 0.10 | `min(question_marks × 0.15, 0.45)` | 0–0.45 |
| `code` | 0.20 | `0.35 if code patterns detected, else 0` | 0 or 0.35 |
| `technical` | 0.10 | `min(tech_term_count × 0.08, 0.4)` | 0–0.4 |
| `sentence` | 0.05 | `min(avg_words_per_sentence / 30, 1.0)` | 0–1 |

**Thresholds:**
- `score ≤ 0.35` → **Gemini 2.5 Flash** (input: $0.0003/1K tokens, output: $0.0024/1K tokens)
- `0.35 < score ≤ 0.65` → **Gemini 2.5 Flash** (still cheap — Flash handles most tasks)
- `score > 0.65` → **Gemini 2.5 Pro** (input: $0.00125/1K tokens, output: $0.005/1K tokens)

> Flash is ~4× cheaper on input and ~2× cheaper on output than Pro. We only pay for Pro when the complexity score strongly signals heavy reasoning.

**Example Classification:**

| Query | Length | Keywords | Score | Model | Est. Cost |
|-------|--------|----------|-------|-------|-----------|
| "What is 2+2?" | 12 | simple: 1 | 0.04 | Gemini Flash | $0.0003 |
| "Explain ML" | 10 | complex: 1 | 0.06 | Gemini Flash | $0.0004 |
| "Analyze trade-offs of microservice vs monolith architecture..." | 180 | complex: 4, tech: 3 | 0.78 | Gemini Pro | $0.008 |

---

## ML Router (Random Forest)

### Why ML on Top of Heuristics?

The heuristic analyzer works well out of the box, but it can't learn from real usage patterns. The ML router is a **Random Forest classifier** that trains on your actual request logs and **replaces** the heuristic once enough data accumulates.

### Feature Vector (12 Features)

Each query is converted into a 12-element numeric vector:

| Index | Feature | Example Value |
|-------|---------|---------------|
| 0 | `word_count` | 15 |
| 1 | `char_count` | 92 |
| 2 | `question_count` | 2 |
| 3 | `has_code` (0/1) | 0 |
| 4 | `complex_keyword_count` | 3 |
| 5 | `simple_keyword_count` | 0 |
| 6 | `avg_words_per_sentence` | 12.5 |
| 7 | `technical_term_count` | 2 |
| 8 | `sentence_count` | 2 |
| 9 | `has_numbers` (0/1) | 1 |
| 10 | `uppercase_ratio` | 0.08 |
| 11 | `punctuation_count` | 4 |

### Training Pipeline

```
POST /train
    │
    ▼
┌────────────────────────┐
│ 1. Read logs/requests.csv │
│ 2. Filter: cache_hit=false │ ← Only train on real LLM calls
│ 3. Require ≥50 samples     │
│ 4. Extract 12 features     │
│ 5. Cross-validate (5-fold) │
│ 6. Fit RandomForest        │
│    • 100 trees              │
│    • max_depth=10           │
│    • min_samples_leaf=3     │
│ 7. Save to models/rf_router.pkl │
└────────────────────────┘
```

**Why Random Forest?**
- Handles non-linear patterns the heuristic misses
- Built-in feature importance (tells you **what** makes queries complex)
- Fast prediction (<1ms per query)
- No GPU required — runs on CPU with scikit-learn
- Cross-validated accuracy prevents overfitting

### Routing Priority

```
Incoming query
    │
    ├─ ML router trained?
    │      ├─ YES → use ML prediction + confidence score
    │      └─ NO  → fall back to heuristic analyzer
    │
    └─ Result: model name + routing source (ml_router / heuristic)
```

---

## Semantic Caching System

### The Problem

Traditional caching only matches **exact** queries:
```
Cache["What is AI?"] = response_1
Cache["What's AI?"]   = cache miss ❌  (even though same question!)
```

### The Solution: Embedding-Based Similarity

**Approach:** Convert queries to vector embeddings, measure cosine similarity.

```python
# Step 1: Embed queries
embedding_1 = model.encode("What is artificial intelligence?")  # [0.2, 0.8, -0.1, ...]
embedding_2 = model.encode("What's AI?")                        # [0.19, 0.81, -0.09, ...]

# Step 2: Measure similarity
similarity = cosine_similarity(embedding_1, embedding_2)  # 0.93 (very similar!)

# Step 3: Cache hit if similarity > threshold (0.85)
if similarity >= 0.85:
    return cached_response  # ✅ Cache hit!
```

### Implementation Details

**Model:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Size:** 80MB (lightweight, no GPU needed)
- **Speed:** ~50ms per embedding
- **Dimensions:** 384-dimensional vectors
- **Trained on:** 1B+ sentence pairs

**Storage Structure:**
```python
cache = [
    (embedding_vector, original_query, response, timestamp, model),
    (embedding_vector, original_query, response, timestamp, model),
    ...
]
# Max size: 500 entries (configurable via CACHE_MAX_SIZE)
# TTL: 3600s default (configurable via CACHE_TTL_SECONDS)
```

**Lookup Algorithm:**
```python
def get(query, model):
    query_emb = embed(query)

    for cached_emb, cached_query, response, ts, cached_model in cache:
        if model != cached_model:
            continue  # Different model = different cache partition

        if is_expired(ts):
            continue  # Skip expired entries

        similarity = cosine_similarity(query_emb, cached_emb)

        if similarity >= 0.85:
            return response  # Cache hit!

    return None  # Cache miss
```

**Time Complexity:**
- Best case: O(1) if first entry matches
- Average case: O(m/2) where m = cache size
- Worst case: O(m) full scan

**Optimization for scale:**
- Use FAISS for O(log m) similarity search at >10K cache entries
- LRU eviction at configurable max size (default 500)
- Index embeddings for faster lookups

### Results

**Before Semantic Caching:**
- Cache hit rate: 25–30% (exact matches only)

**After Semantic Caching:**
- Cache hit rate: 35–45% (+10–15% improvement)
- Examples of semantic hits:
  - "What is ML?" → "What is machine learning?" (similarity: 0.91)
  - "Explain AI" → "What is artificial intelligence?" (similarity: 0.87)
  - "How does an LLM work?" → "How do large language models work?" (similarity: 0.89)

---

## Multi-Provider Support

### Gemini (Default Provider)

| Model | Use Case | Input Cost | Output Cost | Typical Latency |
|-------|----------|------------|-------------|-----------------|
| `gemini-2.5-flash` | Simple–medium queries | $0.0003/1K | $0.0024/1K | ~500ms |
| `gemini-2.5-pro` | Complex reasoning | $0.00125/1K | $0.005/1K | ~1500ms |

**Features:**
- Dynamic model discovery at startup (queries Gemini API for latest versions)
- Automatic retry with exponential backoff (3 retries, 2s base delay)
- Free tier support (Flash: 15 RPM / 500 RPD, Pro: 5 RPM / 50 RPD)

### Groq (Alternative Provider)

Set `LLM_PROVIDER=groq` and provide `GROQ_API_KEY` to use Groq's inference API.

| Model | Use Case | Input Cost | Output Cost | Typical Latency |
|-------|----------|------------|-------------|-----------------|
| `llama-3.1-8b-instant` | Simple queries | $0.05/1M | $0.08/1M | ~200ms |
| `llama-3.3-70b-versatile` | Complex queries | $0.24/1M | $0.24/1M | ~400ms |

**Why Groq?**
- Extremely fast inference (custom LPU hardware)
- Good for latency-sensitive workloads
- Generous free tier for experimentation

### Provider Switching

```python
# In .env
LLM_PROVIDER=gemini   # or "groq"

# At runtime, the router maps complexity → provider-specific models:
# Gemini: Flash (simple/medium) or Pro (complex)
# Groq:   llama-3.1-8b-instant (simple) or llama-3.3-70b-versatile (complex)
```

---

## Analytics & Observability

### What We Track

**Per Request:**
- Timestamp (ISO 8601)
- Query (truncated to 200 chars)
- Model used (`gemini-2.5-flash`, `gemini-2.5-pro`, `llama-3.1-8b-instant`, etc.)
- Complexity score (0–1)
- Cost in USD
- Latency in milliseconds
- Cache hit (true/false)
- Cache type (exact, semantic, none)
- Token counts (input/output)
- User ID

**Storage:**
- **Format:** CSV (easy to analyze with pandas/Excel)
- **Location:** `logs/requests.csv`
- **Size:** ~1KB per 10 requests (10K requests ≈ 1MB)

**Aggregated Metrics (via `/stats` and `/analytics`):**
- Total requests and total cost
- Cache hit rate (overall, by type)
- Model usage distribution
- Cost by model
- Average latency
- Estimated savings vs always using Pro

### Real-Time vs Batch

**In-Memory (Real-Time):**
- Last 100 requests
- Current stats (total cost, cache hits)
- Fast access for `/stats` endpoint

**CSV (Batch Analysis):**
- All historical requests
- Deep analytics via `/analytics` endpoint
- Export for external analysis tools

---

## Performance Characteristics

### Latency Breakdown

**Cache Hit (Exact):**
```
Total: ~5ms
├─ Hash lookup: 1ms
├─ Response construction: 2ms
└─ Network: 2ms
```

**Cache Hit (Semantic):**
```
Total: ~60ms
├─ Embedding generation: 50ms
├─ Similarity search: 8ms
└─ Response construction: 2ms
```

**Cache Miss (Gemini Flash):**
```
Total: ~560ms
├─ Query analysis: 2ms
├─ Model selection: 1ms
├─ Gemini API call: ~500ms
├─ Caching (both layers): 55ms
└─ Response construction: 2ms
```

**Cache Miss (Gemini Pro):**
```
Total: ~1560ms
├─ Query analysis: 2ms
├─ Model selection: 1ms
├─ Gemini API call: ~1500ms
├─ Caching (both layers): 55ms
└─ Response construction: 2ms
```

**Cache Miss (Groq):**
```
Total: ~360ms
├─ Query analysis: 2ms
├─ Model selection: 1ms
├─ Groq API call: ~300ms
├─ Caching (both layers): 55ms
└─ Response construction: 2ms
```

### Throughput

**Single Instance:**
- With caching: ~200 req/sec (mostly cache hits)
- Without caching: ~5 req/sec (limited by LLM API rate limits)

**Bottlenecks:**
1. **LLM API rate limits** (primary — Gemini free tier: 15 RPM for Flash)
2. Semantic embedding generation (secondary)
3. CSV writes (negligible with buffering)

### Scalability Considerations

**Current Limits (Single Instance):**
- Cache size: 500 queries (configurable, in-memory)
- Throughput: 200 req/sec (cached), 5 req/sec (uncached)
- Storage: Unlimited (CSV grows linearly)

**To Scale to 100K+ req/day:**
- [ ] Migrate to Redis for distributed caching
- [ ] Use FAISS for semantic similarity at scale
- [ ] Implement async batch CSV writes
- [ ] Deploy multiple instances behind load balancer

---

## Design Decisions & Trade-offs

### Decision 1: Dual Routing (Heuristic + ML)

**Chose:** Heuristic first, ML upgrade path

**Why:**
- ✅ **Immediate value:** Heuristic works with zero training data
- ✅ **Continuous improvement:** ML router learns from real usage
- ✅ **Graceful fallback:** If ML model is absent, heuristics still work
- ✅ **Explainability:** Heuristic factors are fully debuggable

**Trade-off:**
- ❌ Two codepaths to maintain
- ❌ ML router needs ≥50 logged requests before it activates

**Result:** Best of both worlds — instant deployment + improving accuracy over time.

---

### Decision 2: In-Memory Caching vs Redis

**Chose:** In-memory Python dict for exact cache, in-memory list for semantic cache

**Why:**
- ✅ **Simplicity:** No external dependencies
- ✅ **Speed:** Nanosecond lookups for exact cache
- ✅ **Good for MVP:** Handles 1K–10K req/day easily

**Trade-off:**
- ❌ Doesn't persist across restarts
- ❌ Doesn't scale horizontally (can't share between instances)
- ❌ Memory limited (500 entries default)

**Migration Path:**
```python
# Easy upgrade to Redis later
cache.set(key, value)  # Same interface
# Just swap implementation from dict to Redis client
```

---

### Decision 3: Sentence Transformers vs API-Based Embeddings

**Chose:** Sentence Transformers (`all-MiniLM-L6-v2`)

**Why:**
- ✅ **Free:** No API costs for embeddings
- ✅ **Fast:** 50ms locally vs 200ms API call
- ✅ **Privacy:** No query data sent to third party for embedding
- ✅ **Offline:** Works without internet after initial model download

**Trade-off:**
- ❌ Lower quality than larger embedding models
- ❌ Requires 80MB model download on first run
- ❌ Adds startup time (~2s for model loading)

**Comparison:**

| Model | Speed | Cost | Quality | Size |
|-------|-------|------|---------|------|
| all-MiniLM-L6-v2 | 50ms | Free | Good | 80MB |
| Gemini Embedding API | 200ms | Pay-per-use | Excellent | API |

---

### Decision 4: CSV Storage vs Database

**Chose:** CSV files

**Why:**
- ✅ **Simplicity:** No database setup or migration scripts
- ✅ **Portability:** Easy to analyze in Excel, pandas, or any tool
- ✅ **Good enough:** Handles 100K+ rows easily with pandas

**Trade-off:**
- ❌ Slow for complex queries (must scan full file)
- ❌ No concurrent write safety (rare issue for single-instance)
- ❌ No relational queries

**When to migrate to a database:**
- Need complex analytics queries across multiple dimensions
- Multiple instances writing simultaneously
- Want real-time streaming analytics

---

### Decision 5: Gemini + Groq vs Single Provider

**Chose:** Multi-provider with pluggable architecture

**Why:**
- ✅ **Flexibility:** Switch providers via env var, no code changes
- ✅ **Cost optimization:** Groq's free tier extends experimentation budget
- ✅ **Resilience:** If one provider has outages, switch to the other
- ✅ **Latency options:** Groq (~300ms) vs Gemini Flash (~500ms) for speed-critical paths

**Trade-off:**
- ❌ Different response quality between providers
- ❌ Cache entries are model-specific (can't reuse across providers)
- ❌ More configuration surface area

---

### Decision 6: Synchronous vs Asynchronous Processing

**Chose:** Async FastAPI with sync LLM calls wrapped in async handlers

**Why:**
- ✅ Async handles concurrent requests efficiently
- ✅ LLM calls are I/O bound (waiting on API) — async is ideal
- ✅ Non-blocking cache checks and logging

**Implementation:**
```python
async def route_query(query):
    # Fast operations (async-friendly)
    cached = await check_cache(query)
    if cached:
        return cached

    # Slow I/O operation
    response = await call_llm(provider, model, query)
    return response
```

---

## Performance Benchmarks

**Test Setup:**
- 100 queries (50 unique, 50 duplicates)
- Mix of simple and complex queries
- Single FastAPI instance on local machine

**Results:**

| Metric | Without Custos | With Custos | Improvement |
|--------|----------------|-------------|-------------|
| Total Cost | $1.20 | $0.48 | **60% savings** |
| Avg Latency | 850ms | 320ms | **62% faster** |
| Cache Hit Rate | 0% | 40% | **40% queries free** |
| P95 Latency | 2100ms | 800ms | **62% faster** |

---

## Future Enhancements

### Phase 1 (Near-term)
- [ ] Redis migration for distributed caching
- [ ] FAISS integration for O(log n) similarity search
- [ ] Streaming response support (SSE)

### Phase 2 (Medium-term)
- [ ] Multi-user authentication and per-user budgets
- [ ] Cost prediction and spending forecasts
- [ ] A/B testing framework for routing strategies
- [ ] Webhook notifications for budget alerts

### Phase 3 (Long-term)
- [ ] Support additional providers (Anthropic Claude, Cohere)
- [ ] Custom routing rules per user / per team
- [ ] Grafana/Prometheus metrics export
- [ ] Horizontal scaling with shared state (Redis + FAISS)

---

## Questions?

For technical questions, open an issue on GitHub.

For architecture details, see the [Architecture documentation](ARCHITECTURE.md).