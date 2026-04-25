# 🍜 Yelp Multi-Agent Review Prediction System

A **CrewAI Flows + Crews** multi-agent system that predicts Yelp star ratings and generates realistic review text, given a `user_id` and `business_id`. Built for the **AgentSociety Challenge — Milestone 1**.

---

## 🏗️ Architecture Overview

```
YelpRecommendationFlow (main.py)
  │
  ├─ [1] initialize_request        — log user/item IDs
  ├─ [2] fetch_user_profile        — Crew: analyze_user_task
  ├─ [3] fetch_item_profile        — wait 45 s → Crew: analyze_item_task
  ├─ [4] fetch_web_research        — wait 30 s → Crew: web_research_task
  └─ [5] run_final_prediction      — wait 60 s → Crew: predict_review_task
        └─ [6] process_and_save_results — parse JSON → save report
```

Each stage is an **independent Crew kickoff** with its own agent subset, retry logic, and rate-limit buffer. Results are threaded through a typed Pydantic `FlowState`.

---

## 👥 Agents

| Agent | Role | Tools | allow_delegation |
|---|---|---|---|
| `user_analyst` | Yelp User Profiler | `lookup_user_by_id`, `lookup_reviews_by_user_id`, `user_rag_tool`, `review_rag_tool` | collaborative mode only |
| `item_analyst` | Yelp Restaurant Analyst | `lookup_item_by_id`, `lookup_reviews_by_business_id`, `item_rag_tool`, `review_rag_tool` | collaborative mode only |
| `prediction_modeler` | Review Prediction Expert | *(none — synthesis only)* | False |
| `manager` | Project Manager | delegation + ask tools (built-in) | True |
| `web_researcher` | External Trend Researcher | `SerperDevTool` (if `SERPER_API_KEY` set) | False |

**Process modes** (set via `PROCESS_TYPE` env var):
- `hierarchical` — Manager-led; manager dynamically delegates to workers
- `sequential` / `collaborative` — Workers run tasks directly in sequence

---

## 📋 Tasks

| Task | Agent | Key Inputs | Output |
|---|---|---|---|
| `analyze_user_task` | `user_analyst` | `{user_id}` | Markdown: name, elite years, compliments, avg stars, taste profile |
| `analyze_item_task` | `item_analyst` | `{item_id}` | Markdown: name, address, attributes, hours, customer sentiment |
| `web_research_task` | `web_researcher` | `{item_context}` (extracts business name) | 3–5 sentence web summary |
| `predict_review_task` | `prediction_modeler` | `{user_context}`, `{item_context}`, `{web_context}` | **`{"stars": float, "review": string}`** |

---

## 🧠 Knowledge System

Two `StringKnowledgeSource` objects are injected globally at **Crew level**, visible to every agent via automatic retrieval before each task:

| Source | File | Purpose |
|---|---|---|
| Yelp Schema | `docs/Yelp Data Translation.md` | Prevents hallucination on fields like `compliment_hot`, `elite`, `RestaurantsPriceRange2` |
| EDA Findings | `docs/eda_findings.md` | Provides dataset-level statistics to calibrate predictions |

**Embedding model:** `BAAI/bge-small-en-v1.5` (HuggingFace sentence-transformer, CPU-compatible, no API cost)  
**Collection name:** `yelp_knowledge_v3` (ChromaDB)

---

## 🔧 Tools

### Exact Lookup Tools (Primary — Python `@tool`)

These **guarantee exact data** by scanning JSON files line-by-line for matching IDs:

| Tool | Input | Data Source | Notes |
|---|---|---|---|
| `lookup_user_by_id` | `user_id` string | `data/user_subset.json` | Strips `friends` list to save tokens |
| `lookup_item_by_id` | `item_id` string | `data/item_subset.json` | Matches on `business_id` or `item_id` |
| `lookup_reviews_by_user_id` | `user_id` string | `data/review_subset.json` | Returns up to 10 reviews |
| `lookup_reviews_by_business_id` | `business_id` string | `data/review_subset.json` | Returns up to 10 reviews |

### RAG Tools (Fallback — `JSONSearchTool` + ChromaDB)

Three semantic search tools over ChromaDB vector collections:

| Tool | Collection | When to use |
|---|---|---|
| `search_user_profile_data` | `benchmark_true_fresh_index_Filtered_User_1` | Broad queries: "users who review Vietnamese food" |
| `search_restaurant_feature_data` | `benchmark_true_fresh_index_Filtered_Item_1` | Category/feature searches |
| `search_historical_reviews_data` | `benchmark_true_fresh_index_Filtered_Review_1` | Semantic review search |

> ⚠️ **RAG cannot reliably find a specific user or business by ID.** See the [RAG Problem section](#-the-rag-hallucination-problem) below and check(#-RAG_IMPLEMENTATION_PROOF) for more details.

---

## ⚠️ The RAG Hallucination Problem

### What went wrong

Using `JSONSearchTool` to look up a specific user by ID (e.g., `_BcWyKQL16ndpBdggh2kNA`) triggers a **semantic similarity search** over vector embeddings. Yelp IDs are random base64-like strings — they have no semantic meaning. The search returns random unrelated user records.

The agent then **hallucinated** a confident but completely wrong answer:
```json
{"user_id": "_BcWyKQL16ndpBdggh2kNA", "username": "Sara", "review_count": 15}
```

**Actual data:** Karen, 4274 reviews, avg 3.69★, Elite since 2008.

### The fix: exact lookup tools first

All four `lookup_*` tools were implemented as Python functions that do a deterministic **line-by-line scan** for an exact ID match. Agent backstories and task descriptions explicitly warn agents to use lookup tools for specific IDs and treat RAG as semantic-only fallback.

### Smart ChromaDB cache detection

`create_rag_tool()` in `crew.py` uses native `sqlite3` to inspect `chroma.sqlite3` before initializing any RAG tool:

```python
cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
if cursor.fetchone() is not None:
    collection_exists = True
```

- **Collection exists** → Instant load (<1 second). The Pydantic schema is forced to `FixedJSONSearchToolSchema` to hide `json_path` from the agent, preventing accidental 1–4 hour re-indexing.
- **Collection missing** → Normal initialization with `json_path` triggers a one-time full indexing run.

---

## 🌊 Flow Design Details

### State management

```python
class YelpRecommendationState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    user_profile: str = ""    # analyze_user_task output
    item_profile: str = ""    # analyze_item_task output
    web_research: str = ""    # web_research_task output
    raw_result: str = ""      # predict_review_task raw output
    final_report: dict = {}   # parsed JSON
```

### Retry logic

Every task runs through `_run_with_retries()` with up to 3 attempts:
- Backoff: 30 s after attempt 1, 60 s after attempt 2
- Checks for degraded output markers: `"Too Many Requests"`, `"Timeout Error"`, `"mentioned not found"`, etc.

### Rate limiting between steps

Configurable via environment variables:

| Env Variable | Default | Stage |
|---|---|---|
| `FLOW_STAGGER_SECONDS` | 45 | Between user and item analysis |
| `FLOW_WEB_WAIT_SECONDS` | 30 | Before web research |
| `FLOW_PREDICTION_WAIT_SECONDS` | 60 | Before final prediction |

### Context truncation

Before the final prediction, contexts are truncated to avoid token budget overruns:
- `user_context`: `PREDICTION_MAX_CTX_CHARS` characters (default 1200)
- `item_context`: 1200 characters
- `web_context`: 600 characters

### Hierarchical → Collaborative fallback

If the final prediction fails in hierarchical mode, the Flow automatically retries in collaborative mode, then restores the original setting.

### JSON extraction (multi-layer)

`extract_json_from_output()` handles messy LLM output:
1. Strip markdown code fences
2. Regex `{...}` search + `json.loads()`
3. Manual regex field extraction as last resort

---

## 📦 Prerequisites

| Item | Requirement |
|---|---|
| Python | 3.12+ |
| Package Manager | [Astral `uv`](https://docs.astral.sh/uv/) |
| OS | macOS / Linux / Windows |
| LLM API | NVIDIA Build API Key (free) **or** local Ollama |
| Web Search | Serper API Key (optional) |

---


## 📂 Data Files

| File | Description |
|---|---|
| `data/user_subset.json` | User profiles (review_count, avg_stars, elite, compliments, etc.) |
| `data/item_subset.json` | Business profiles (name, address, categories, attributes, hours) |
| `data/review_subset.json` | Historical review texts (user_id, business_id, stars, text) |
| `data/test_review_subset.json` | Test cases (user_id + item_id pairs to predict) |

**ChromaDB Collections:**

| Collection | Embedding Source |
|---|---|
| `benchmark_true_fresh_index_Filtered_User_1` | `data/user_subset.json` |
| `benchmark_true_fresh_index_Filtered_Item_1` | `data/item_subset.json` |
| `benchmark_true_fresh_index_Filtered_Review_1` | `data/review_subset.json` |

**Embedding model:** `BAAI/bge-small-en-v1.5` (384-dimensional, CPU)

---

## 📁 Project Structure

```
Rag_Crew_Profiler/
├── src/
│   └── first_crew/
│       ├── crew.py              ← Agent/Task/Tool definitions + RAG setup
│       ├── main.py              ← YelpRecommendationFlow (entry point)
│       ├── config/
│       │   ├── agents.yaml      ← Agent roles, goals, backstories
│       │   └── tasks.yaml       ← Task descriptions and expected outputs
│       ├── benchmark_rag.py     ← RAG retrieval speed test
│       └── benchmark_indexing.py ← Full re-indexing script (1–4 hours)
├── data/
│   ├── user_subset.json
│   ├── item_subset.json
│   ├── review_subset.json
│   └── test_review_subset.json
├── docs/
│   ├── Yelp Data Translation.md ← Schema definitions (global knowledge)
│   └── eda_findings.md          ← EDA statistics (global knowledge)
├── .env                         ← API keys and config (not committed)
├── pyproject.toml
└── README.md
```

---

## 🧪 Milestone 1 Test Result

**Input:** First record from `data/test_review_subset.json`
- User: `_BcWyKQL16ndpBdggh2kNA` (Karen — Elite 2008–2021, avg 3.69★, 4274 reviews, 558 fans)
- Business: `uBDXcXlLR9IuRV1N2m0SPQ` (Pho Street, 2104 Market St, Philadelphia PA)

**Web research finding:** The Market St location is listed as **CLOSED** on Yelp; a new location exists at 1230 Arch St.

**Prediction output:**
```json
{
  "stars": 4.0,
  "review": "4.0 stars - I stopped by Pho Street recently, and I must say, their pho really hit the spot. The broth was rich and flavorful, and the service was friendly and attentive. I've been to my fair share of Vietnamese places, but Pho Street stands out for its consistency and quality. The casual atmosphere makes it a great spot for a quick lunch or dinner, and the delivery option is a bonus. While it didn't blow me away, Pho Street is definitely a solid choice for anyone craving a good bowl of pho. Recommended!"
}
```

**Rationale:** Karen's 3.69★ average + Pho Street's 4.0★ overall rating + positive web sentiment → 4.0★ predicted.


---

## 📚 References

- [CrewAI Documentation](https://docs.crewai.com/en/introduction)
- [CrewAI Flows](https://docs.crewai.com/en/concepts/flows)
- [CrewAI Hierarchical Process](https://docs.crewai.com/en/learn/hierarchical-process)
- [AgentSociety Challenge](https://github.com/tsinghua-fib-lab/AgentSocietyChallenge)
- [NVIDIA Build API](https://build.nvidia.com/)
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
