# ML Router — Design & Implementation

> A Random Forest classifier that learns from real usage patterns to improve routing decisions over the heuristic analyzer.

---

## 1. Overview

Custos routes LLM queries to either a cheap model (`gemini-2.5-flash`) or an expensive model (`gemini-2.5-pro`) based on query complexity. The **heuristic analyzer** handles this from day one using weighted keyword/structure signals. The **ML router** is a second-stage system that learns from accumulated request logs to improve routing accuracy over time.

### Why Two Systems?

| | Heuristic Analyzer | ML Router |
|---|---|---|
| **Works from day one** | ✅ Zero training data needed | ❌ Requires ≥50 logged requests |
| **Deterministic** | ✅ Same input → same output | ❌ Depends on training data |
| **Handles edge cases** | ❌ Misses non-obvious patterns | ✅ Learns from real query distribution |
| **Debuggable** | ✅ Inspect each weighted signal | ⚠️ Feature importances help, but less transparent |

The ML router does **not** replace the heuristic permanently. It only overrides the heuristic when its confidence exceeds 0.7, falling back to heuristic routing otherwise. This confidence gating prevents the ML model from making low-confidence mistakes.

---

## 2. Feature Vector

Every query is converted to a **12-element numeric vector** before classification. The features are extracted in [`_extract_features()`](file:///d:/research/llm-cost-firewall/app/ml_router.py#L62-L81).

| # | Feature | Description | `"What is Python?"` | `"Analyze the trade-offs between microservice and monolithic architectures for a high-traffic e-commerce platform, considering latency, team structure, and deployment complexity"` |
|---|---|---|---|---|
| 0 | `word_count` | Number of whitespace-separated tokens | 3 | 22 |
| 1 | `char_count` | Total character length | 16 | 165 |
| 2 | `question_count` | Number of `?` characters | 1 | 0 |
| 3 | `has_code` | 1.0 if code markers detected (`` ``` ``, `def `, `class `, `import `, `function(`) | 0.0 | 0.0 |
| 4 | `complex_keyword_count` | Hits against 30+ complex keywords (`analyze`, `compare`, `trade-offs`, `architect`, ...) | 0 | 3 (`analyze`, `trade-offs`, `architect`) |
| 5 | `simple_keyword_count` | Hits against 17 simple keywords (`what is`, `define`, `list`, ...) | 1 (`what is`) | 0 |
| 6 | `avg_words_per_sentence` | Mean word count per sentence (split on `.!?`) | 3.0 | 22.0 |
| 7 | `tech_term_count` | Regex matches against technical terms (`api`, `architecture`, `microservice`, `latency`, ...) | 0 | 3 (`architecture`, `microservice`, `latency`) |
| 8 | `sentence_count` | Number of non-empty sentences | 1 | 1 |
| 9 | `has_numbers` | 1.0 if any digit `\d+` found | 0.0 | 0.0 |
| 10 | `uppercase_ratio` | Fraction of uppercase characters | 0.0625 | 0.006 |
| 11 | `punctuation_count` | Count of `.,;:!?` characters | 1 | 3 |

### Feature Design Rationale

- **Word/char counts** capture query length, a strong baseline signal (longer queries tend to be more complex).
- **Keyword counts** mirror the heuristic's strongest signal (weight 0.40) but give the Random Forest freedom to learn non-linear combinations.
- **`has_code`** is binary because code presence alone triggers a need for a capable model, regardless of code length.
- **`avg_words_per_sentence`** captures structural complexity — multi-clause sentences correlate with analytical reasoning.
- **`uppercase_ratio`** catches edge cases like all-caps queries or acronym-heavy technical prompts.

---

## 3. Training

### When to Train

Training is triggered manually via the `POST /train` endpoint. It is **not** automatic to avoid retraining on small incremental data or during high-traffic periods.

```bash
curl -X POST http://localhost:8000/train
```

### Requirements

| Requirement | Value | Why |
|---|---|---|
| Minimum samples | 50 non-cached requests | Below this, the model overfits and cross-validation fails |
| Data source | `logs/requests.csv` | The CSV log written by `RequestLogger` |
| Label column | `model_used` | The model that the **heuristic** selected for each request |
| Filter | `cache_hit == "false"` only | Cached requests don't have real model selections |

### Algorithm Configuration

```python
RandomForestClassifier(
    n_estimators=100,       # 100 decision trees in the ensemble
    max_depth=10,           # prevent overfitting on small datasets
    min_samples_leaf=3,     # each leaf must represent ≥3 real queries
    random_state=42,        # reproducible results across training runs
)
```

### Cross-Validation

Before fitting the final model, the training pipeline runs **k-fold cross-validation** to estimate real-world accuracy:

```python
cv_folds = min(5, len(df) // 10)
cv_scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
accuracy = cv_scores.mean()
```

- `k` is capped at 5 folds, but reduced for smaller datasets (e.g., 50 samples → 5 folds of 10).
- The reported accuracy is the **mean across all folds**, not the training accuracy. This prevents optimistic bias.

### Training Response

A successful training call returns:

```json
{
  "success": true,
  "samples_used": 127,
  "cv_accuracy": 0.891,
  "classes": ["gemini-2.5-flash", "gemini-2.5-pro"],
  "top_features": {
    "complex_keywords": 0.2341,
    "word_count": 0.1876,
    "char_count": 0.1543,
    "tech_terms": 0.1102,
    "avg_words_sentence": 0.0897
  }
}
```

If insufficient data:

```json
{
  "success": false,
  "reason": "Need ≥50 real requests to train. Have 23.",
  "have": 23,
  "need": 50
}
```

---

## 4. Prediction

Once trained, the ML router is consulted on every incoming request before the heuristic result is used.

### Prediction Flow

```
                         ┌──────────────────┐
           query ───────►│  _extract_features │
                         │  (12-dim vector)   │
                         └────────┬───────────┘
                                  │
                         ┌────────▼───────────┐
                         │  clf.predict_proba  │
                         │  → [0.23, 0.77]     │
                         └────────┬───────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  confidence = max(proba)    │
                    │  = 0.77                     │
                    └─────────────┬──────────────┘
                                  │
                         ┌────────▼───────────┐
                         │  confidence > 0.7?  │
                         └──┬─────────────┬───┘
                         YES│             │NO
                   ┌────────▼──────┐  ┌───▼──────────────┐
                   │  Use ML model │  │  Use heuristic   │
                   │  prediction   │  │  result instead   │
                   └───────────────┘  └──────────────────┘
```

### Confidence Gating

The 0.7 confidence threshold is critical. The router returns `predict_proba` — the fraction of trees that voted for the winning class:

```python
ml_result = self._ml_router.predict(query)
model = (
    ml_result["model"]
    if ml_result["confidence"] > 0.7
    else select_model(complexity_score)
)
```

**Why 0.7?** Below this threshold, the Random Forest is uncertain — roughly 30%+ of trees disagree. In that ambiguous zone, the deterministic heuristic is safer than a coin-flip ML decision. The 0.7 value was chosen empirically; it can be tuned.

### Fallback Behavior

If the model is not trained (`is_trained == False`), `predict()` returns immediately:

```python
{"model": "gemini-2.5-flash", "confidence": 0.0, "source": "heuristic_fallback"}
```

The zero confidence ensures the caller always falls through to the heuristic. The system behaves identically to a deployment without scikit-learn.

---

## 5. Feature Importances

After training, the `POST /train` endpoint returns the **top-5 most important features**, ranked by Gini importance (mean decrease in impurity across all 100 trees).

Example output:

```
Feature              Importance
──────────────────────────────
complex_keywords      0.2341    ← strongest signal
word_count            0.1876
char_count            0.1543
tech_terms            0.1102
avg_words_sentence    0.0897
```

### How to Interpret

- **`complex_keywords` dominance** is expected — the heuristic weights keywords at 0.40, and the ML model independently discovers the same signal.
- **`word_count` and `char_count`** being high confirms that query length is a strong proxy for complexity.
- If `has_code` ranks highly, it means the training data included many code-analysis queries, skewing importance.
- Low-importance features (e.g., `has_numbers`, `uppercase_ratio`) are retained because they occasionally help edge cases without adding noise (Random Forest handles irrelevant features gracefully).

### Using Importances to Debug

If the ML router performs poorly, check feature importances:
- If a single feature dominates (>0.5), the model may be overfitting to one signal.
- If `simple_keywords` is high, the training data may be skewed toward simple queries.
- Retrain with more diverse queries to rebalance.

---

## 6. Model Persistence

The trained model is serialized to disk so it survives server restarts.

| Parameter | Value |
|---|---|
| Save path | `models/rf_router.pkl` |
| Format | Python `pickle` |
| Contents | `{"clf": RandomForestClassifier, "classes": [...], "accuracy": float, "n_samples": int}` |
| Auto-load | On startup, `MLRouter.__init__()` calls `_load()` |

### Lifecycle

```
Server starts → MLRouter.__init__() → _load()
                                        ↓
                          models/rf_router.pkl exists?
                          ├── YES → deserialize, is_trained = True
                          └── NO  → is_trained = False, use heuristic
                          
POST /train → train() → _save()
                          ↓
                models/rf_router.pkl written to disk
```

### Retraining

Calling `POST /train` again overwrites the saved model. There is no model versioning — the latest training run always wins. To roll back, delete `models/rf_router.pkl` and restart the server (reverts to heuristic-only).

---

## 7. Graceful Degradation

The ML router is designed to work **without** its ML dependencies installed. This enables deployment on minimal environments.

```python
try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
```

### Behavior Without Dependencies

| Scenario | Behavior |
|---|---|
| `scikit-learn` not installed | `ML_AVAILABLE = False`, training returns `{"success": false, "reason": "scikit-learn not installed"}` |
| `numpy` not installed | Same as above (sklearn depends on numpy) |
| `pandas` not installed | Same (used to read CSV logs) |
| Dependencies installed, no training data | `is_trained = False`, all predictions fall back to heuristic |
| Dependencies installed, <50 samples | Training rejected with `"Need ≥50 real requests"` |

The application always starts and serves requests. The ML router is a **progressive enhancement**, not a requirement.

### Required Packages for ML Features

```
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

These are included in `requirements.txt` but excluded from `requirements-render.txt` (production deploy with minimal footprint).

---

## 8. Why Random Forest?

The choice of Random Forest over other classifiers was deliberate:

### Evaluated Alternatives

| Algorithm | Pros | Cons | Why Not |
|---|---|---|---|
| **Logistic Regression** | Fast, interpretable, linear | Can't capture non-linear keyword interactions | Keyword combinations matter (e.g., "analyze" + "code" is different from just "analyze") |
| **SVM** | Strong with small datasets | Less interpretable, slower predict | No probability calibration by default, harder to explain |
| **Neural Network** | Captures complex patterns | Needs GPU, large data, slow to train | Overkill for 12 features and <10K samples |
| **XGBoost** | Higher accuracy potential | Extra dependency, more hyperparameters | Marginal accuracy gain not worth the complexity |
| **Random Forest** | ✅ Non-linear, interpretable, fast | Slightly less accurate than boosted methods | Selected |

### Key Advantages for This Use Case

1. **No hyperparameter sensitivity.** RF works well out of the box. The defaults (`n_estimators=100`, `max_depth=10`) are robust. XGBoost requires careful tuning of learning rate, depth, and regularization.

2. **Built-in feature importance.** The Gini importance ranking tells the user *what makes queries complex*, which is valuable for debugging and understanding the system. This is harder with SVMs or neural nets.

3. **Fast prediction.** A single prediction traverses 100 shallow trees in <1 ms. This adds negligible latency to the request pipeline. Neural networks would add 10–100 ms.

4. **Handles irrelevant features gracefully.** If `has_numbers` turns out to be useless, RF simply doesn't split on it. No need for feature selection or regularization.

5. **Works with small data.** 50–500 training samples is sufficient. Neural networks need 10,000+. RF's ensemble averaging prevents overfitting on small datasets, and `min_samples_leaf=3` adds further regularization.

6. **Probabilistic output.** `predict_proba` returns per-class probabilities, enabling the confidence gating mechanism. The fraction-of-trees interpretation is intuitive: "77% of trees voted for Pro."

### What Could Be Better

- **Boosted methods (XGBoost, LightGBM)** would likely achieve 2–5% higher accuracy at scale (>1,000 samples), but add a dependency and tuning burden.
- **Online learning** (e.g., incremental models) would allow the model to update continuously without manual retraining. Currently, retraining is manual via `POST /train`.
- **Feature engineering** could be expanded — e.g., bigram counts, query embedding similarity to known complex queries. The current 12 features are intentionally simple.

---

## 9. End-to-End Example

### Step 1: Accumulate Data

Send 50+ real queries through Custos:

```bash
for i in $(seq 1 60); do
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"Sample query number $i about various topics\"}"
done
```

### Step 2: Train

```bash
curl -X POST http://localhost:8000/train
```

Response:
```json
{
  "success": true,
  "samples_used": 58,
  "cv_accuracy": 0.862,
  "classes": ["gemini-2.5-flash", "gemini-2.5-pro"],
  "top_features": {
    "complex_keywords": 0.2341,
    "word_count": 0.1876,
    "char_count": 0.1543,
    "tech_terms": 0.1102,
    "avg_words_sentence": 0.0897
  }
}
```

### Step 3: Verify

The ML router is now active. Subsequent requests include ML-based routing when confidence > 0.7:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

The `/stats` endpoint confirms the ML router is active:

```json
{
  "ml_router": {
    "trained": true,
    "model_type": "RandomForest"
  }
}
```

### Step 4: Reset (if needed)

Delete the saved model and restart:

```bash
rm models/rf_router.pkl
# restart server
```

---

*Implementation: [ml_router.py](file:///d:/research/llm-cost-firewall/app/ml_router.py) | Training data: [logger.py](file:///d:/research/llm-cost-firewall/app/logger.py) | Integration: [router.py](file:///d:/research/llm-cost-firewall/app/router.py#L168-L176)*
