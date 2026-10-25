# Commodity Forecaster - Data Dictionary

This document details the raw dataset schemas, data types, missing value distributions, and data quality characteristics for the Commodity Price Forecasting system.

---

## 1. Overview & File Inventory

The raw data consists of 4 Microsoft Excel (`.xlsx`) workbooks located in `backend/data/raw/`, sourced from the **Ministry of Agriculture & Farmers Welfare** (AgMarknet) and the **Department of Consumer Affairs (DOCA)**.

| File Name | File Size | Sheet Name | Total Rows (incl. headers) | Data Rows | Date Range |
|---|---|---|---|---|---|
| `onion_andhra_pradesh.xlsx` | 914,460 bytes | `Data Sheet` | 2,484 | 2,480 | 01 Jan 2020 – 15 Oct 2026 |
| `onion_karnataka.xlsx` | 927,130 bytes | `Data Sheet` | 2,484 | 2,480 | 01 Jan 2020 – 15 Oct 2026 |
| `onion_madhya_pradesh.xlsx` | 935,871 bytes | `Data Sheet` | 2,484 | 2,480 | 01 Jan 2020 – 15 Oct 2026 |
| `onion_maharashtra.xlsx` | 934,091 bytes | `Data Sheet` | 2,484 | 2,480 | 01 Jan 2020 – 15 Oct 2026 |

---

## 2. File Structure & Header Layout

All 4 workbooks share an identical layout:

1. **Header / Metadata Rows (Rows 0 to 2)**:
   - The first 3 rows contain title and source metadata.
   - **Ingestion Rule**: Ingestion parsers (e.g. `pandas.read_excel`) must skip the first 3 rows (`skiprows=3` or `header=3`).
2. **Column Names Row (Row 3)**:
   - Contains raw header titles. Note: Column 0 is labeled `Ministry of Agriculture & Farmers Welfare`, but data rows contain the state name.
3. **Data Records (Row 4 to 2,483)**:
   - 2,480 consecutive daily calendar records spanning from `01 Jan 2020` to `15 Oct 2026`.

---

## 3. Schema & Column Specifications

| Col Index | Raw Header Name | Canonical Field Name | Raw Type | Target Type | Units | Description & Source |
|---|---|---|---|---|---|---|
| 0 | `Ministry of Agriculture & Farmers Welfare` | `state` | object | string | Categorical | Name of the Indian state (`Andhra Pradesh`, `Karnataka`, `Madhya Pradesh`, `Maharashtra`). |
| 1 | `Crop` | `crop` | object | string | Categorical | Commodity name (Constant: `Onion`). |
| 2 | `Calendar/Date` | `date` | object | date (`YYYY-MM-DD`) | Daily date | Observation date in `'DD Mon YYYY'` format (e.g., `'01 Jan 2020'`). |
| 3 | `Mandi Arrival Quantity (AgMarknet)` | `arrival_quantity` | object | float64 | Tonnes | Aggregated wholesale market arrival volume reported via AgMarknet. Serves as a key supply factor/covariate. |
| 4 | `Mandi Modal Price (AgMarknet)` | `modal_price` | object | float64 | Rs/Qtl | Wholesale modal price (most frequent transaction price) in mandis reported by AgMarknet. Rs per Quintal (1 Quintal = 100 kg). Primary wholesale price forecast target. |
| 5 | `Retail Price (DOCA)` | `retail_price` | object | float64 | Rs/Qtl | Retail price reported by the Department of Consumer Affairs (DOCA) across state reporting centers. Rs per Quintal. Primary retail price forecast target. |

---

## 4. State-Wise Null & Quality Analysis

Each file contains 2,480 daily records. Missing values vary significantly by state and reporting agency:

### 4.1. Andhra Pradesh (`onion_andhra_pradesh.xlsx`)
- **Total Records**: 2,480
- **Mandi Arrival Quantity**: 733 nulls (29.56%)
- **Mandi Modal Price**: 733 nulls (29.56%)
- **Retail Price**: 15 nulls (0.60%)
- **Notes**: High missingness in wholesale mandi reporting. Modal price and arrival null counts match exactly, pointing to non-trading days or missing APMC aggregate reporting. Retail price reporting from DOCA is very complete.

### 4.2. Karnataka (`onion_karnataka.xlsx`)
- **Total Records**: 2,480
- **Mandi Arrival Quantity**: 121 nulls (4.88%)
- **Mandi Modal Price**: 121 nulls (4.88%)
- **Retail Price**: 113 nulls (4.56%)
- **Notes**: Moderate and consistent coverage across both wholesale (AgMarknet) and retail (DOCA) series.

### 4.3. Madhya Pradesh (`onion_madhya_pradesh.xlsx`)
- **Total Records**: 2,480
- **Mandi Arrival Quantity**: 19 nulls (0.77%)
- **Mandi Modal Price**: 19 nulls (0.77%)
- **Retail Price**: 0 nulls (0.00%)
- **Notes**: Exceptionally clean and complete dataset with zero missing retail price observations and <1% missing wholesale data.

### 4.4. Maharashtra (`onion_maharashtra.xlsx`)
- **Total Records**: 2,480
- **Mandi Arrival Quantity**: 39 nulls (1.57%)
- **Mandi Modal Price**: 39 nulls (1.57%)
- **Retail Price**: 5 nulls (0.20%)
- **Notes**: As the primary onion producing state in India, Maharashtra features very high reporting consistency (<2% nulls in wholesale, 5 nulls in retail). Benchmark state for baseline models.

---

## 5. Ingestion & Transformation Guidelines

1. **Type Parsing**: All numeric columns (`arrival_quantity`, `modal_price`, `retail_price`) are stored as strings/objects in Excel. Ingestion pipelines must strip formatting, convert blanks to `NaN` / `NULL`, and cast to `float64`.
2. **Date Parsing**: Date strings formatted as `'DD Mon YYYY'` must be parsed into ISO-8601 calendar dates (`YYYY-MM-DD`).
3. **Missing Value Strategy**:
   - For time series modeling (e.g. Chronos, ARIMA/ETS baselines), missing values should be handled via forward fill (`ffill`), linear interpolation, or calendar reindexing depending on whether non-trading days should be modeled explicitly.
4. **Target Variables**:
   - `modal_price`: Mandi wholesale benchmark.
   - `retail_price`: End-consumer benchmark.
5. **Covariates**:
   - `arrival_quantity`: Wholesale supply volume directly influencing price equilibrium.
