# 🧅 AgriCast: Commodity Price Forecasting Agent

AgriCast is an enterprise-grade AI chat agent designed for agricultural commodity price analytics and multi-horizon probabilistic forecasting. Built strictly following the 11-phase architecture specification, the agent combines **DuckDB**, **Amazon Chronos-2**, **LangGraph**, and **FastAPI** to provide market intelligence with price risk alerts and arrival covariates.

Currently populated with 6+ years of daily AgMarknet and DOCA onion market data (Jan 2020 – Oct 2026 across Maharashtra, Karnataka, Andhra Pradesh, and Madhya Pradesh), the system is fully configuration-driven: adding a new commodity requires YAML configuration changes, zero code modifications.

---

## 🏛️ Architecture & Key Design Principles

The codebase strictly enforces clean layered architecture:
```
[ Frontend Demo (Streamlit) ]
              │  HTTP JSON only
              ▼
[ FastAPI API Layer (/api/v1) ]
              │
[ LangGraph Agent & Multi-Layer Guardrails ]
              │
[ Pydantic Tool Registry (Envelopes & UI Blocks) ]
              │
[ Core Services (Pipeline, Chronos Forecaster, Risk Checker, Cache) ]
              │
[ Semantic Layer & Parameterized Data Access (DuckDB) ]
```

### Strict Architectural Invariants:
1. **Zero SQL/Code by LLM**: The LLM never generates or executes SQL or code. It selects tools and populates validated Pydantic parameters.
2. **Zero Raw Rows to LLM**: Tools return opaque handles (`data_ref`, `forecast_id`) plus bounded summary statistics.
3. **Number Grounding**: Every numeric assertion emitted by the LLM is strictly validated against tool results by the Layer 4 output guard.
4. **Deterministic Off-Topic Guardrail**: Off-topic queries (e.g., entertainment, politics, general coding) are rejected immediately prior to any LLM or tool invocation.
5. **Decoupled Backend & Frontend**: The backend contains no UI dependencies (`backend/` has zero Streamlit/HTML imports). The frontend talks strictly to the HTTP API contract (`/api/v1`).
6. **No Hardcoded Commodities**: All series, units, filters, and thresholds are defined in `semantic_layer.yaml` and `thresholds.yaml`.

---

## 📁 Repository Structure

```
commodity-forecaster/
├── backend/
│   ├── app/
│   │   ├── agent/                 # LangGraph workflow, state, prompts, guardrails
│   │   │   ├── prompts/           # System prompt & scope classifier
│   │   │   ├── graph.py           # StateGraph orchestration & checkpointing
│   │   │   ├── guardrails.py      # Input, scope, output grounding, rate limits
│   │   │   └── state.py           # AgentState Pydantic model
│   │   ├── api/                   # FastAPI routes & Pydantic request/response schemas
│   │   ├── config/                # YAML configurations
│   │   │   ├── semantic_layer.yaml# Single source of truth for series & commodities
│   │   │   ├── thresholds.yaml    # MSP, danger, critical price risk thresholds
│   │   │   └── settings.yaml      # Model names, backtest holdout, tolerance
│   │   ├── core/                  # Pure Python core services (no UI, no LLM)
│   │   │   ├── blocks.py          # Pre-built UI block builders (charts, alerts, summary)
│   │   │   ├── cache.py           # In-memory LRU store for data_ref & forecast_id
│   │   │   ├── data_access.py     # Parameterized DuckDB SQL execution
│   │   │   ├── forecaster.py      # Chronos-2 wrapper & multi-quantile engine
│   │   │   ├── models.py          # ToolResult envelope & UI block schemas
│   │   │   ├── pipeline.py        # Cleaning, regularizing, gap filling, IQR outlier
│   │   │   ├── risk.py            # Threshold evaluator & risk alert generator
│   │   │   ├── semantic_layer.py  # Catalog lookup, synonym resolution
│   │   │   └── settings.py        # Hierarchical settings loader
│   │   ├── tools/                 # Validated Pydantic tool wrappers
│   │   └── main.py                # FastAPI application, CORS, singleton lifecycle
│   ├── data/
│   │   ├── duckdb/                # Persistent DuckDB database (commodities.duckdb)
│   │   └── raw/                   # Raw AgMarknet / DOCA Excel files
│   ├── scripts/
│   │   └── ingest.py              # Normalized data ingestion into DuckDB
│   ├── tests/                     # Comprehensive pytest test suite (Phase 1–9)
│   ├── requirements.txt
│   ├── DATA_DICTIONARY.md
│   └── DECISIONS.md
├── frontend_demo/
│   ├── app.py                     # Streamlit interactive UI calling FastAPI
│   └── requirements.txt
└── README.md
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

Ensure Python 3.11+ is installed. Clone or navigate to the project directory:

```bash
cd commodity-forecaster
```

Install backend dependencies:
```bash
pip install -r backend/requirements.txt
```

Install frontend dependencies:
```bash
pip install -r frontend_demo/requirements.txt
```

### 2. Configure Environment Variables (`.env`)

Copy `.env.example` in `backend/` to `.env`:
```bash
cp backend/.env.example backend/.env
```

Paste your Groq API key:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
```

> **Note on Offline Fallback**: If `GROQ_API_KEY` is not provided, the application automatically activates `FallbackOfflineLLM`, allowing local testing and end-to-end demonstrations without external network access.

### 3. Data Ingestion into DuckDB

To re-ingest raw Excel files from `backend/data/raw/` into `backend/data/duckdb/commodities.duckdb`:
```bash
python backend/scripts/ingest.py
```
*Output: Normalizes dates, cleans columns, removes negative prices, and indexes 9,920 daily records across 4 major producing states.*

### 4. Run Backend API Server

Start the FastAPI backend with Uvicorn:
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`
- System Health: `http://localhost:8000/api/v1/health`

### 5. Run Streamlit Frontend Demo

In a separate terminal:
```bash
cd frontend_demo
streamlit run app.py --server.port 8501
```
Open `http://localhost:8501` in your browser.

---

## 📊 Core Features & Capabilities

### 1. Multi-Horizon Forecasting with Amazon Chronos-2
- Forecast horizons: **7, 15, 30, and 90 days** (or any integer 1–90).
- Emits **p10, p50, and p90** probabilistic quantile forecasts.
- Automated holdout backtesting computing **MAPE** and **MAE** metrics.
- Multi-horizon caveats attached automatically for projections $\ge 60$ days.

### 2. Covariate-Informed Forecasts
- Mandi arrival quantities stitched and synchronized as leading indicators.
- Automatic side-by-side comparison between baseline unconditioned forecast and arrival-conditioned forecast.

### 3. Threshold-Driven Risk Warnings
- Evaluates forecasted price trajectories against configured thresholds (`critical_high`, `high`, `moderate_high`, `critical_low`).
- Generates structured risk alert cards showing threshold value, expected crossing date, and actionable advisories.

### 4. Interactive Visualizations
- Multi-quantile fan charts with shaded uncertainty bands (`forecast_band`).
- Historical time-series exploration (`line`).
- Period-over-period comparative analytics (`line_compare`).
- Grouped seasonal and monthly aggregations (`bar`).

---

## 🧪 Running the Test Suite

Run all verification tests across all phases:
```bash
cd backend
python -m pytest tests -v
```

Test coverage includes:
- `test_phase1.py`: Ingestion, schema validation, semantic layer resolution.
- `test_phase2.py`: DataAccess, catalog, series, compare, aggregate tools.
- `test_phase3.py`: Cleaning, gap filling, outlier detection, data integrity.
- `test_phase4.py`: Chronos multi-quantile inference, backtesting, horizons.
- `test_phase5.py`: Threshold configuration, risk evaluation, crossing dates.
- `test_phase6.py`: Covariate stitching, forecast tool, baseline vs factor.
- `test_phase7.py`: Guardrails (off-topic rejection, grounding, rate limits).
- `test_phase8.py`: LangGraph state machine, cycles, checkpointing.
- `test_phase9.py`: FastAPI endpoints, error handling, contract responses.
