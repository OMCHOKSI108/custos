# Custos

[![Live API](https://img.shields.io/badge/API-custos--lqtf.onrender.com-gold?logo=render)](https://custos-lqtf.onrender.com/docs)
[![Tests](https://github.com/OMCHOKSI108/custos/actions/workflows/test.yml/badge.svg)](https://github.com/OMCHOKSI108/custos/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![Deployed on Render](https://img.shields.io/badge/deployed-Render-purple.svg)](https://render.com)

An intelligent LLM proxy that reduces Gemini API costs through smart routing, dual-layer semantic caching, budget enforcement, and rate limiting.

**Live API:** `https://custos-lqtf.onrender.com/docs`

---

## Overview

Every LLM-powered product treats all queries the same. A simple factual question hits the same expensive model as a multi-paragraph architecture analysis. Custos fixes that by analyzing each query's complexity and routing it to the most cost-effective model.

> Custos achieved an average cost reduction of 60% across 100 test queries through intelligent routing and semantic caching. Simple queries routed to the cheaper model saved up to 90% per request.

---

## Architecture

```mermaid
graph TD
    Client["Your Application"] -->|POST /chat| RL["Rate Limiter"]
    RL -->|"Sliding Window"| BE["Budget Enforcer"]
    BE -->|"HTTP 402 on limit"| CA["Complexity Analyzer"]
    CA -->|"6-factor scoring"| MR["ML Router<br/>Random Forest"]
    MR -->|"prediction"| EC["Exact Cache<br/>SHA-256"]
    EC -->|"miss"| SC["Semantic Cache<br/>all-MiniLM-L6-v2"]
    SC -->|"hit > 0.85"| Resp["Response"]
    EC -->|"hit"| Resp
    SC -->|"miss"| GC["Gemini API"]
    GC -->|"gemini-2.5-flash"| Resp
    GC -->|"gemini-2.5-pro"| Resp

    style Client fill:#1a1a2e,stroke:#e94560,color:#fff
    style Resp fill:#1a1a2e,stroke:#e94560,color:#fff
    style GC fill:#16213e,stroke:#0f3460,color:#fff
    style RL fill:#1b4332,stroke:#40916c,color:#fff
    style BE fill:#1b4332,stroke:#40916c,color:#fff
    style CA fill:#3d2b1f,stroke:#b8860b,color:#fff
    style MR fill:#3d2b1f,stroke:#b8860b,color:#fff
    style EC fill:#2d1b69,stroke:#7b2d8e,color:#fff
    style SC fill:#2d1b69,stroke:#7b2d8e,color:#fff
```

The request pipeline consists of six sequential stages. A query first passes through rate limiting and budget enforcement. It then undergoes complexity analysis using six weighted signals. The ML router selects the optimal model based on historical training data. Before reaching the Gemini API, the system checks two caching layers: an exact SHA-256 hash cache and a semantic similarity cache using sentence-transformers embeddings with a cosine similarity threshold.

![Architecture Diagram](docs/architecture.png)

> The semantic cache handles paraphrased queries. Asking "What is machine learning?" and "Explain ML" returns the same cached response, eliminating redundant API calls and reducing latency by up to 60 milliseconds per hit.

---

## Features

**Complexity Analysis**

The analyzer scores queries from 0.0 to 1.0 using six weighted signals. Length accounts for 15 percent of the score, keyword matching for 40 percent, question count for 10 percent, code presence detection for 20 percent, technical jargon density for 10 percent, and sentence structure for 5 percent. Queries scoring below 0.35 route to gemini-2.5-flash while scores above 0.65 trigger gemini-2.5-pro.

**Dual Caching**

The caching system operates in two tiers. The exact cache generates a SHA-256 hash of each normalized query and returns a match in constant time. The semantic cache converts queries to 384-dimensional embeddings using all-MiniLM-L6-v2 and compares against stored entries using cosine similarity. Hits above the 0.85 threshold return the cached response. This combination increases cache hit rates from 25-30 percent to 35-45 percent, with each semantic hit saving approximately 50 milliseconds in embedding generation plus the full API call time.

**Budget Enforcement**

The enforcer tracks spending in daily and hourly buckets. Each request checks current spending against configurable limits and returns HTTP 402 when exceeded. Limits can be updated at runtime through the /budget/configure endpoint without restarting the server.

**Rate Limiting**

The sliding window rate limiter maintains a deque of timestamps per user, removing entries older than one hour before checking against the limit. This approach prevents burst abuse at window boundaries, a known weakness of fixed-window counters.

**Cost Prediction**

The /predict endpoint estimates the expense of a query before making an actual API call. It computes the complexity score, selects the target model, applies the model's per-token pricing, and returns an estimated cost value.

---

## Quick Start

Clone the repository and set up a virtual environment.

```
git clone https://github.com/OMCHOKSI108/custos.git
cd custos
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a .env file from the example template and add your Gemini API key. Keys are available at aistudio.google.com without requiring a credit card.

```
cp .env.example .env
# Edit .env and add your API key(s)
```

Launch the server with uvicorn and open the interactive API documentation at http://localhost:8000/docs.

```
uvicorn app.main:app --reload
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| /chat | POST | Send a query and receive a routed LLM response |
| /stats | GET | Retrieve live metrics for cache, budget, routing, and latency |
| /predict | GET | Estimate cost for a query without making an API call |
| /compare | GET | Compare projected costs across all available models |
| /history | GET | View request log filtered by user_id |
| /export/csv | GET | Download complete request history as CSV |
| /train | POST | Train the ML router on accumulated request logs |
| /budget/configure | POST | Update spending limits at runtime |
| /cache | DELETE | Clear both exact and semantic caches |

---
## Testing

The test suite covers 73 tests across 6 modules using pytest.

```bash
python -m pytest tests/ -v
```

---

## How It Works

```mermaid
flowchart LR
    U["You have an existing project<br/>with LLM costs going high"] --> G["You find Custos on GitHub<br/>OMCHOKSI108/custos"]
    G --> K{"Choose your path"}
    K -->|"Zero setup"| A["Use our hosted API<br/>custos-lqtf.onrender.com"]
    K -->|"Self-host"| B["Deploy your own server<br/>with your API keys"]
    K -->|"Import in code"| C["pip install + import Custos<br/>in your Python project"]
    A --> D["Send POST /chat<br/>with query + user_id"]
    B --> D
    C --> E["client = Custos(api_key)<br/>client.query('...')"]
    D --> F["Custos routes smartly<br/>cheap model for simple queries"]
    E --> F
    F --> H["You save ~60% on LLM costs<br/>+ get caching + budget control"]
    H --> I["You are impressed<br/>and tell others about Custos"]
    
    style U fill:#1e3a5f,stroke:#f59e0b,color:#fff
    style G fill:#1e3a5f,stroke:#f59e0b,color:#fff
    style H fill:#1a4731,stroke:#34d399,color:#fff
    style I fill:#1a4731,stroke:#34d399,color:#fff
```

---

## Integration

You can integrate Custos into your project in several ways.

### Option 1: Use the Hosted API (Zero Setup)

The live API is deployed at `https://custos-lqtf.onrender.com`. Just send HTTP requests from any language.

```python
import requests

resp = requests.post(
    "https://custos-lqtf.onrender.com/chat",
    json={"query": "What is Python?", "user_id": "user_1"},
    timeout=60,
).json()

print(resp["response"])
print("Cost:", resp["cost_usd"], "| Model:", resp["model"])
```

```javascript
const resp = await fetch("https://custos-lqtf.onrender.com/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "Explain ML", user_id: "dev" }),
});
const data = await resp.json();
console.log(data.response, data.cost_usd);
```

```bash
curl -X POST https://custos-lqtf.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain transformers","user_id":"dev"}'
```

**Note:** Render free tier sleeps after 15 minutes of inactivity. The first request after idle takes ~30 seconds to wake up, then runs at full speed.

### Option 2: As a Python Package (Direct Import)

Clone or copy the `app/` folder into your project, then import the `Custos` client class.

```python
from app import Custos

client = Custos(
    gemini_api_key="AIzaSy...",
    provider="gemini",
    daily_budget=10.0,
    hourly_budget=2.0,
    rate_limit=100,
)

response = client.query(
    query="What is Python?",
    user_id="user_1",
)
print(response["response"])
print("Model used:", response["model"])
print("Cost:", response["cost_usd"])
print("Cache hit:", response["cache_hit"])
```

Switch providers per-query:

```python
response = client.query("Explain ML", provider="groq")
response = client.query("Explain ML", provider="gemini")
```

Access analytics and cache control:

```python
stats = client.stats()
print("Total requests:", stats["analytics"]["total_requests"])
print("Cache hit rate:", stats["analytics"]["cache_hit_rate_pct"], "%")

client.clear_cache()
```

### Option 3: As a Proxy Server (Self-Hosted)

Run Custos as a standalone service and send HTTP requests from any language.

```bash
# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "query": "What is machine learning?",
        "user_id": "my_app_user",
    },
).json()

print(response["response"])
print("Cost:", response["cost_usd"])
```

```javascript
// JavaScript / Node.js
const response = await fetch("http://localhost:8000/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ query: "What is Python?", user_id: "user_1" }),
});
const data = await response.json();
console.log(data.response, data.cost_usd);
```

```bash
# curl
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Explain transformers","user_id":"user_1"}'
```

### Option 4: As a Docker Service

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t custos .
docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY=your_key \
  -e MOCK_MODE=false \
  custos
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| GEMINI_API_KEY | "" | Google Gemini API key |
| GROQ_API_KEY | "" | Groq API key (alternative provider) |
| LLM_PROVIDER | "gemini" | Default provider: "gemini" or "groq" |
| MOCK_MODE | "true" | Set "false" to use real API calls |
| DAILY_BUDGET_USD | 10.0 | Maximum daily spend |
| HOURLY_BUDGET_USD | 2.0 | Maximum hourly spend |
| RATE_LIMIT_PER_HOUR | 100 | Max requests per user per hour |
| CACHE_TTL_SECONDS | 3600 | Cache expiration time |

---

## Screenshots

**API Documentation** — Interactive Swagger UI at /docs

![Swagger API Documentation](docs/swagger.png)

**Live Dashboard** — Real-time monitoring with cache stats, budget tracking, and latency metrics

![Custos Dashboard](docs/Dashboard.png)

**Request History** — Detailed log of all queries with timestamps, costs, and caching status

![Request History](docs/history.png)

**Query Response** — Example output showing cost breakdown, model selection, and caching status

![Sample Output](docs/Output1.png)

---

## Project Structure

```
custos/
  app/
    main.py           FastAPI application with all HTTP endpoints
    router.py         Six-layer pipeline orchestrator
    analyzer.py       Complexity scorer using six weighted signals
    cache.py          Exact match cache with LRU eviction and TTL
    semantic_cache.py Embedding similarity cache with cosine threshold
    budget.py         Daily and hourly spending tracker with enforcement
    rate_limiter.py   Per-user sliding window rate limiter
    ml_router.py      Random Forest classifier trained on request logs
    analytics.py      Pandas-based statistics aggregation
    logger.py         CSV-based request logging
    config.py         Environment variable configuration
    auth.py           Optional API key authentication middleware
  tests/              Unit and integration test suite
  frontend/dashboard/ Browser-based dashboard with Chart.js
  scripts/            Benchmark and utility scripts
  docs/               Technical documentation and screenshots
  render.yaml         Render.com deployment configuration
  Dockerfile          Production Docker image
  docker-compose.yml  Docker Compose setup
  .env.example        Environment variable template
```

---

## Limitations

- **In-memory state**: Caches, rate limit windows, and budget buckets are stored in memory. They don't persist across restarts and can't be shared between instances.
- **CSV logging**: Request logs are stored in CSV files. This works for single-instance deployments but isn't suitable for high-concurrency production with multiple workers.
- **Semantic cache linear scan**: The semantic cache does a brute-force linear scan over all entries (O(n)). This is fast up to ~500 entries but would need FAISS or similar for larger deployments.
- **Single-instance**: There is no distributed lock or shared state mechanism. Deploying multiple instances would result in separate caches and budget trackers.
- **Heuristic routing**: The default complexity analyzer uses keyword matching and heuristics. It works well for common queries but can misroute edge cases. The ML router improves accuracy once trained.

---

## Future Improvements

- **Redis integration** for distributed caching, rate limiting, and budget enforcement across multiple instances
- **PostgreSQL/SQLite** for persistent, queryable request logging and analytics
- **FAISS** for scalable semantic similarity search at >1K cached entries
- **Multi-provider A/B testing** to compare model quality vs cost across providers
- **Custom routing rules per user** (e.g., always use Pro for user X)
- **Prometheus + Grafana** for production-grade observability
- **Streaming responses** for real-time token delivery
- **Request queuing** with priority-based scheduling during rate limit windows

---

## Author

**OMCHOKSI**

GitHub: https://github.com/OMCHOKSI108
