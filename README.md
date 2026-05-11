# AI Brand Visibility Tracker

A Python system that tracks how often a target brand appears in AI-generated answers across multiple LLMs (OpenAI, Anthropic, Gemini), analyzes sentiment and citation sources, and recommends which content channels to invest in to improve brand visibility.

Inspired by tools like [Peec AI](https://peec.ai) — built with real APIs instead of UI scraping for educational and portfolio purposes.

---

## Architecture

```mermaid
flowchart TD
    A([Prompt Configs\n40 prompts × 4 categories]) --> B

    subgraph M1["Module 1 — Prompt Runner"]
        B[asyncio.gather]
        B --> C1[OpenAI\ngpt-4o-mini]
        B --> C2[Anthropic\nclaude-haiku-4-5]
        B --> C3[Gemini\n2.5-flash]
        B --> C4[MockClient\nauto-fallback]
        C1 & C2 & C3 & C4 --> V[vote.py\nmajority voting n=3]
    end

    V --> D[(Raw JSON\ndata/raw/)]

    subgraph M2["Module 2 — Brand & Source Analyzer"]
        D --> E1[Brand Detector\nregex + spaCy NER]
        D --> E2[Sentiment Judge\nLLM-as-judge gpt-4o-mini]
        D --> E3[Citation Extractor\nURL regex + domain classifier]
    end

    E1 & E2 & E3 --> F

    subgraph M3["Module 3 — Visibility Metrics"]
        F[store.py\nSQLite / BigQuery]
        F --> G1[(BigQuery\nbrand_mentions\ncitation_sources)]
        G1 --> G2[calculator.py\nvisibility % · position trend\nsentiment moving avg]
        G2 --> G3[Plotly Dash\nDashboard :8050]
    end

    G1 --> H

    subgraph M4["Module 4 — Recommendation Engine"]
        H[feature_engineer.py\nco-occurrence features]
        H --> I1[LightGBM\nR²=0.80]
        H --> I2[Rule-based scorer\nfallback]
        I1 & I2 --> J[FastAPI :8000\n/recommendations\n/retrain 🔒\n/visibility]
    end

    style M1 fill:#dbeafe,stroke:#3b82f6
    style M2 fill:#dcfce7,stroke:#22c55e
    style M3 fill:#fef9c3,stroke:#eab308
    style M4 fill:#fce7f3,stroke:#ec4899
```

### Module 1 — Prompt Runner
Queries multiple LLM APIs concurrently using `asyncio.gather`. Supports OpenAI, Anthropic, and Gemini. Automatically falls back to a `MockClient` when an API key is missing or invalid.

Includes **majority voting** (`vote.py`): run each prompt N times and keep only brands that appear in more than half the trials. Eliminates false positives caused by LLM non-determinism.

### Module 2 — Brand & Source Analyzer
Parses LLM responses to extract:
- **Brand mentions** — which brands appear, in what position order, with what sentiment
- **Sentiment** — LLM-as-judge via `gpt-4o-mini` (batched async calls)
- **Citation sources** — URLs extracted via regex, classified into 10 domain types (review_site, tech_media, community, etc.)

### Module 3 — Visibility Metrics
Persists results to BigQuery (production) or SQLite (local dev) with a unified interface. Computes time-series metrics: visibility %, position trend, sentiment moving average, competitor gap. Visualized with a Plotly Dash dashboard.

### Module 4 — Recommendation Engine
Trains a LightGBM model on citation co-occurrence features to predict opportunity scores per domain type. Served via FastAPI. `/retrain` requires an API key and reloads the model in memory without server restart.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| LLM APIs | OpenAI API, Anthropic API, Google Gemini API |
| Concurrency | `asyncio`, `tenacity` (retry) |
| NLP | spaCy NER, LLM-as-judge (gpt-4o-mini) |
| ML | LightGBM, scikit-learn |
| Storage | Google BigQuery, SQLite |
| Visualization | Plotly Dash |
| API | FastAPI, Uvicorn |
| Data | pandas, Pydantic |
| Testing | pytest, pytest-asyncio |
| Infra | Docker, python-dotenv |

---

## Project Structure

```
ai-brand-visibility-tracker/
├── src/
│   ├── prompt_runner/
│   │   ├── llm_clients.py      # OpenAI / Anthropic / Gemini / Mock clients
│   │   ├── runner.py           # asyncio concurrent query engine (n_runs support)
│   │   └── vote.py             # majority voting across N trials
│   ├── analyzer/
│   │   ├── brand_detector.py   # regex + spaCy NER brand matching
│   │   ├── sentiment_judge.py  # LLM-as-judge sentiment scoring
│   │   ├── citation_extractor.py # URL extraction + domain classification
│   │   └── pipeline.py         # orchestrates all analyzer steps
│   ├── metrics/
│   │   ├── calculator.py       # visibility %, position trend, competitor gap
│   │   └── dashboard.py        # Plotly Dash 4-chart dashboard
│   ├── storage/
│   │   ├── store.py            # unified interface (auto-routes BQ vs SQLite)
│   │   ├── bigquery_store.py   # BigQuery backend
│   │   ├── sqlite_store.py     # SQLite backend
│   │   └── schema.py           # shared DDL / BQ schema definitions
│   ├── recommender/
│   │   ├── feature_engineer.py # builds feature matrix from BQ data
│   │   ├── scorer.py           # rule-based opportunity scorer (fallback)
│   │   ├── train_lgbm.py       # LightGBM training + inference
│   │   └── api.py              # FastAPI endpoints
│   └── utils/
│       ├── config.py           # env vars + paths
│       └── models.py           # Pydantic data models
├── tests/
│   ├── test_brand_detector.py  # 13 tests: regex, word boundary, position order
│   ├── test_citation_extractor.py # 16 tests: domain classification, URL parsing
│   ├── test_pipeline.py        # 8 tests: async pipeline with mocked LLM judge
│   ├── test_scorer.py          # 10 tests: opportunity score calculation
│   ├── test_retrain_auth.py    # 4 tests: API key auth on /retrain
│   └── test_vote.py            # 10 tests: majority voting logic
├── data/
│   ├── prompts_batch.py        # 40 prompts across 4 SaaS categories
│   └── raw/                    # raw LLM response JSONs (gitignored)
├── demo_module1.py             # run LLM queries
├── demo_module2.py             # analyze latest run
├── demo_module3.py             # save to DB + launch dashboard
├── demo_module4.py             # launch FastAPI
├── demo_lgbm.py                # train LightGBM + show results
├── run_batch_pipeline.py       # batch data generation (40 prompts × 3 LLMs)
├── pytest.ini
├── Dockerfile
└── environment.yml
```

---

## Setup

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- API keys: OpenAI (required), Anthropic + Gemini (optional — auto-mock if missing)
- GCP project with BigQuery enabled (optional — auto-falls back to SQLite)

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate brand-tracker
python -m spacy download en_core_web_sm
```

> **Apple Silicon (M1/M2/M3):** LightGBM requires the conda-forge build:
> ```bash
> pip uninstall lightgbm -y
> conda install -c conda-forge lightgbm -y
> ```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
OPENAI_API_KEY=sk-proj-...

# Optional — system uses MockClient if empty
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=AIzaSy...

# Optional — system uses SQLite if not configured
GCP_PROJECT_ID=your-project-id
BIGQUERY_DATASET=brand_tracker
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Required to call POST /retrain — set a strong random string
RETRAIN_API_KEY=your-secret-key
```

### 3. (Optional) GCP / BigQuery setup

1. Enable BigQuery API in [GCP Console](https://console.cloud.google.com)
2. Create a Service Account with roles `BigQuery Data Editor` + `BigQuery Job User`
3. Download the JSON key → save to `~/.gcp/brand-tracker-sa.json`
4. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`

---

## Usage

### Quick start — run all modules in order

```bash
# Step 1: Query LLMs
python demo_module1.py

# Step 2: Analyze brand mentions + citations
python demo_module2.py

# Step 3: Save to DB + open dashboard (http://localhost:8050)
python demo_module3.py

# Step 4: Launch recommendation API (http://localhost:8000/docs)
python demo_module4.py
```

### Generate training data (for LightGBM)

```bash
# Dry run — preview 40 prompts without API calls
python run_batch_pipeline.py --dry-run

# Run all 40 prompts × 3 LLMs (~$0.08, ~5-10 min)
python run_batch_pipeline.py

# Run one category only (cheaper test)
python run_batch_pipeline.py --category crm

# Run with majority voting — 3 trials per prompt for stable brand detection
python run_batch_pipeline.py --n-runs 3 --category crm
```

### Train LightGBM model

```bash
python demo_lgbm.py
```

### Run tests

```bash
pytest -v
```

### API endpoints

Once `demo_module4.py` is running:

```bash
# Get recommendations (uses LightGBM if model exists)
curl -X POST http://localhost:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"target_brand": "Asana", "competitors": ["Jira", "Linear", "Monday.com"], "top_n": 5}'

# Visibility summary
curl "http://localhost:8000/visibility?brands=Asana,Jira,Monday.com"

# Competitor gap analysis
curl http://localhost:8000/competitor-gap/Asana

# Retrain model on latest data — requires API key
curl -X POST http://localhost:8000/retrain \
  -H "X-API-Key: your-secret-key"
```

Interactive API docs: `http://localhost:8000/docs`

---

## Data Models

```python
LLMResponse      # run_id, provider, prompt, response_text, latency_ms
BrandMention     # brand, position, sentiment (positive/neutral/negative), snippet
CitationSource   # url, domain, domain_type (review_site / tech_media / community / ...)
```

---

## ML Model

**Problem:** Given a (brand, domain_type) pair, predict opportunity score = how much investing in this source type will improve brand visibility vs competitors.

**Features:**
- `cooccurrence` — how often brand appears in prompts citing this domain type
- `competitor_avg_cooccurrence` — competitor benchmark
- `avg_position_with_source` — brand position quality when cited alongside this source
- `total_mentions` — brand's overall visibility
- `domain_authority_weight` — domain type prior (review sites > social media)
- + 5 more derived features

**Target:** `competitor_avg_cooccurrence - brand_cooccurrence` (positive = gap = opportunity)

**Results (trained on 946 brand_mentions, 919 citations, 25 brands):**

| Metric | Value |
|--------|-------|
| R² | 0.80 |
| MAE | 2.87 |
| Top feature | `cooccurrence` |

---

## Docker

```bash
# Build
docker build -t brand-tracker .

# Train model locally first (if not already done)
python demo_lgbm.py

# Run API server — mount ./data so the container can read the trained model and SQLite DB
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data brand-tracker
```

> **How the model gets loaded:**
> On startup the API tries to load `data/lgbm_opportunity_model.pkl` from the mounted volume.
> If no model file is found (e.g. fresh deployment with BigQuery), it attempts to auto-train from
> available data. If neither is possible it falls back to rule-based scoring.

---

## Data Coverage

Trained and validated across 4 SaaS categories:

| Category | Brands Tracked |
|----------|---------------|
| Project Management | Asana, Jira, Linear, Monday.com, Notion, ClickUp, Trello |
| CRM | HubSpot, Salesforce, Pipedrive, Attio, Zoho CRM, Close |
| AI Writing Tools | Jasper, Copy.ai, Grammarly, Writer, Writesonic, Notion AI |
| Developer Tools | GitHub, GitLab, Linear, Jira, Azure DevOps, Shortcut |
