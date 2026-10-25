# Architectural & Modeling Decisions

This document records the foundational decisions made for the Commodity Price Forecasting Agent backend based on raw data inspection and system requirements.

---

## Decision 1: Scope of Available Data & Covariates

- **Context**: Original project specifications referenced external datasets including `onion_production.csv` (annual/seasonal crop production) and `fuel_prices.csv` (diesel/transportation indices).
- **Finding**: Raw data inspection revealed only four state-level Excel workbooks for Onion from the Ministry of Agriculture & Farmers Welfare / DOCA (`onion_andhra_pradesh.xlsx`, `onion_karnataka.xlsx`, `onion_madhya_pradesh.xlsx`, `onion_maharashtra.xlsx`). No external production or fuel datasets were provided.
- **Decision**: 
  - External references to `onion_production.csv` and `fuel_prices.csv` are discontinued and do not apply.
  - Covariate and factor modeling will leverage `Mandi Arrival Quantity (AgMarknet)` (Tonnes) as the primary supply factor. Wholesale arrivals directly reflect market supply pressure and serve as an endogenous/exogenous covariate for price forecasting.

---

## Decision 2: Geographic Hierarchy & Semantic Layer

- **Context**: The forecasting system needs to support geographic filtering and routing.
- **Finding**: The source data from AgMarknet and DOCA is provided pre-aggregated at the state level. There are no sub-state APMC mandi level breakdowns (e.g. Lasalgaon, Pimpalgaon) or district-level records in the raw files.
- **Decision**:
  - The semantic layer models **States** as the primary geographic dimension rather than individual physical mandis/markets.
  - Default state: **Maharashtra** (the largest onion-producing state in India and the national price discovery anchor).
  - Default market / aggregation: **State-level aggregate**.
  - All SQL queries, tool parameters, and Pydantic models will operate on state-level dimensions (`Andhra Pradesh`, `Karnataka`, `Madhya Pradesh`, `Maharashtra`).

---

## Decision 3: Forecast Targets & Price Series

- **Context**: The raw data includes two distinct price series: `Mandi Modal Price (AgMarknet)` and `Retail Price (DOCA)`.
- **Finding**: 
  - `Mandi Modal Price` represents wholesale mandi transactions (farmer/trader realization).
  - `Retail Price` represents consumer-facing prices tracked by the Department of Consumer Affairs.
- **Decision**:
  - Both series are retained as first-class forecast targets.
  - Tools and models will allow users to forecast either `modal_price` (wholesale) or `retail_price` (retail), with `modal_price` serving as the default wholesale benchmark.

---

## Decision 4: Date Handling, Temporal Reference, & `as_of_date`

- **Context**: Queries often involve relative temporal expressions (e.g., "last month", "next 14 days", "forecast for next week").
- **Finding**: The dataset runs from January 01, 2020 to October 15, 2026.
- **Decision**:
  - The system `as_of_date` is anchored to the latest available date in the dataset: **October 15, 2026** (`2026-10-15`).
  - The LLM agent will always resolve relative date expressions into explicit ISO-8601 calendar strings (`YYYY-MM-DD`) using `as_of_date` before invoking backend tools.
  - Backend tools accept only validated ISO-8601 dates, eliminating date ambiguity in query execution.

---

## Decision 5: Generalization & Zero-Hardcoding Rule

- **Decision**:
  - Although the initial dataset covers Onion, all core abstractions (DuckDB schemas, Pydantic models, data access layers, and tools) will be commodity-agnostic.
  - The string `"onion"` must never be hardcoded into business logic or query generators; instead, commodity names will be parameterized or loaded from configuration.
