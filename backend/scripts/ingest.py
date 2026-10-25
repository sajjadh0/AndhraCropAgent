"""Ingestion script for commodity price datasets into DuckDB.

Reads Excel sheets from raw data directory, normalizes columns, validates
records, and loads them into a centralized DuckDB database.
"""

from datetime import date
import logging
from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


class IngestConfig(BaseModel):
    """Configuration for dataset ingestion."""

    raw_data_dir: Path
    db_path: Path
    table_name: str = "onion_prices"
    sheet_name: str = "Data Sheet"
    skiprows: int = 4


class StateIngestSummary(BaseModel):
    """Summary statistics for ingested state data."""

    state: str
    crop: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    min_date: Optional[date | str] = None
    max_date: Optional[date | str] = None
    null_arrivals: int
    null_modal_prices: int
    null_retail_prices: int


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Excel column names to clean snake_case identifiers.

    Args:
        df: Raw DataFrame loaded with headers.

    Returns:
        DataFrame with standardized column names.
    """
    column_mapping: dict[str, str] = {}
    for col in df.columns:
        col_str = str(col).strip().lower()
        if col_str == "state":
            column_mapping[col] = "state"
        elif col_str == "crop":
            column_mapping[col] = "crop"
        elif "calendar" in col_str or col_str == "date":
            column_mapping[col] = "date"
        elif "arrival" in col_str:
            column_mapping[col] = "arrival_quantity"
        elif "modal" in col_str:
            column_mapping[col] = "modal_price"
        elif "retail" in col_str:
            column_mapping[col] = "retail_price"

    df = df.rename(columns=column_mapping)
    expected_cols = ["state", "crop", "date", "arrival_quantity", "modal_price", "retail_price"]
    available_cols = [c for c in expected_cols if c in df.columns]
    return df[available_cols].copy()


def clean_and_validate(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Parse dates, clean numeric columns, and calculate validity flags.

    Args:
        df: DataFrame with standardized columns.
        source_name: Name of source file for logging context.

    Returns:
        Cleaned DataFrame with parsed types and `is_valid` boolean flag.
    """
    # 1. Clean string fields (forward-fill because Excel only populates row 1)
    if "state" in df.columns:
        df["state"] = df["state"].ffill().astype(str).str.strip()
    if "crop" in df.columns:
        df["crop"] = df["crop"].ffill().astype(str).str.strip()

    # 2. Parse dates ('DD Mon YYYY' format)
    parsed_date = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
    # Fallback for alternative formats if any failed
    if parsed_date.isna().any():
        fallback_date = pd.to_datetime(df["date"], errors="coerce")
        parsed_date = parsed_date.fillna(fallback_date)

    invalid_date_mask = parsed_date.isna()
    if invalid_date_mask.any():
        logger.warning(
            "[%s] Found %d rows with unparseable dates.",
            source_name,
            int(invalid_date_mask.sum()),
        )

    # 3. Clean and convert numeric columns to float64
    numeric_cols = ["arrival_quantity", "modal_price", "retail_price"]
    for col in numeric_cols:
        if col in df.columns:
            # Strip whitespace, commas and coerce errors to NaN
            cleaned_series = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"nan": None, "None": None, "": None})
            )
            df[col] = pd.to_numeric(cleaned_series, errors="coerce").astype("float64")
        else:
            df[col] = pd.Series([None] * len(df), dtype="float64")

    # 4. Check negative values
    negative_modal = df["modal_price"] < 0
    negative_retail = df["retail_price"] < 0
    negative_arrival = df["arrival_quantity"] < 0
    negative_values = negative_modal | negative_retail | negative_arrival

    if negative_values.any():
        logger.warning(
            "[%s] Found %d rows with negative prices or arrivals.",
            source_name,
            int(negative_values.sum()),
        )

    # 5. Determine is_valid flag
    # Row is valid if date is valid AND no negative values
    is_valid_mask = (~invalid_date_mask) & (~negative_values)
    df["is_valid"] = is_valid_mask.astype(bool)

    # Format dates as ISO strings (YYYY-MM-DD) for reliable DuckDB ingestion
    df["date"] = parsed_date.dt.strftime("%Y-%m-%d")

    return df


def load_raw_file(file_path: Path, sheet_name: str, skiprows: int) -> pd.DataFrame:
    """Load and process a single Excel file.

    Args:
        file_path: Path to Excel workbook.
        sheet_name: Name of target worksheet.
        skiprows: Number of header metadata rows to skip.

    Returns:
        Processed DataFrame ready for database insertion.
    """
    logger.info("Reading file: %s", file_path.name)
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
    df_std = standardize_columns(df_raw)
    df_clean = clean_and_validate(df_std, source_name=file_path.name)
    return df_clean


def ingest_data(config: IngestConfig) -> list[StateIngestSummary]:
    """Ingest all Excel files into DuckDB database.

    Args:
        config: Ingestion configuration options.

    Returns:
        List of summary metrics for each state.
    """
    # Ensure raw directory exists
    if not config.raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {config.raw_data_dir}")

    # Ensure output database directory exists
    config.db_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover Excel files
    excel_files = sorted(list(config.raw_data_dir.glob("*.xlsx")))
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {config.raw_data_dir}")

    logger.info("Found %d Excel files to ingest.", len(excel_files))

    # Load and concatenate all files
    frames: list[pd.DataFrame] = []
    for file_path in excel_files:
        df = load_raw_file(file_path, config.sheet_name, config.skiprows)
        frames.append(df)

    combined_df = pd.concat(frames, ignore_index=True)
    logger.info("Total rows read across all files: %d", len(combined_df))

    # Connect to DuckDB
    con = duckdb.connect(database=str(config.db_path))

    try:
        # Idempotent table creation
        con.execute(f"DROP TABLE IF EXISTS {config.table_name}")
        con.execute(
            f"""
            CREATE TABLE {config.table_name} (
                state VARCHAR,
                crop VARCHAR,
                date DATE,
                arrival_quantity DOUBLE,
                modal_price DOUBLE,
                retail_price DOUBLE,
                is_valid BOOLEAN
            )
            """
        )

        # Register dataframe and insert
        con.register("temp_df", combined_df)
        con.execute(
            f"""
            INSERT INTO {config.table_name}
            SELECT
                CAST(state AS VARCHAR),
                CAST(crop AS VARCHAR),
                TRY_CAST(date AS DATE) AS date,
                CAST(arrival_quantity AS DOUBLE),
                CAST(modal_price AS DOUBLE),
                CAST(retail_price AS DOUBLE),
                CAST(is_valid AS BOOLEAN)
            FROM temp_df
            """
        )
        con.unregister("temp_df")

        # Create summary view
        summary_view_name = f"{config.table_name}_summary"
        con.execute(
            f"""
            CREATE OR REPLACE VIEW {summary_view_name} AS
            SELECT
                state,
                crop,
                COUNT(*) AS total_rows,
                COUNT(CASE WHEN is_valid THEN 1 END) AS valid_rows,
                COUNT(CASE WHEN NOT is_valid THEN 1 END) AS invalid_rows,
                MIN(date) AS min_date,
                MAX(date) AS max_date,
                COUNT(CASE WHEN arrival_quantity IS NULL THEN 1 END) AS null_arrivals,
                COUNT(CASE WHEN modal_price IS NULL THEN 1 END) AS null_modal_prices,
                COUNT(CASE WHEN retail_price IS NULL THEN 1 END) AS null_retail_prices
            FROM {config.table_name}
            GROUP BY state, crop
            ORDER BY state
            """
        )

        # Fetch summary data
        summary_query = f"SELECT * FROM {summary_view_name}"
        rows = con.execute(summary_query).fetchall()
        summaries: list[StateIngestSummary] = []
        for r in rows:
            summaries.append(
                StateIngestSummary(
                    state=r[0],
                    crop=r[1],
                    total_rows=r[2],
                    valid_rows=r[3],
                    invalid_rows=r[4],
                    min_date=r[5],
                    max_date=r[6],
                    null_arrivals=r[7],
                    null_modal_prices=r[8],
                    null_retail_prices=r[9],
                )
            )

        return summaries
    finally:
        con.close()


def print_ingest_report(summaries: list[StateIngestSummary], db_path: Path) -> None:
    """Print a clean tabular report of ingestion results.

    Args:
        summaries: List of StateIngestSummary objects.
        db_path: Path to DuckDB database file.
    """
    separator = "=" * 110
    sub_sep = "-" * 110
    print("\n" + separator)
    print("                     COMMODITY PRICE INGESTION REPORT")
    print(f"Database: {db_path.resolve()}")
    print(separator)
    print(
        f"{'State':<20} | {'Crop':<8} | {'Total':<7} | {'Valid':<7} | {'Invalid':<7} | "
        f"{'Date Range':<23} | {'Null Arr':<8} | {'Null Modal':<10} | {'Null Retail':<11}"
    )
    print(sub_sep)

    total_records = 0
    total_valid = 0
    total_invalid = 0

    for s in summaries:
        total_records += s.total_rows
        total_valid += s.valid_rows
        total_invalid += s.invalid_rows
        date_range_str = f"{s.min_date} to {s.max_date}"
        print(
            f"{s.state:<20} | {s.crop:<8} | {s.total_rows:<7} | {s.valid_rows:<7} | {s.invalid_rows:<7} | "
            f"{date_range_str:<23} | {s.null_arrivals:<8} | {s.null_modal_prices:<10} | {s.null_retail_prices:<11}"
        )

    print(sub_sep)
    print(
        f"{'TOTAL':<20} | {'':<8} | {total_records:<7} | {total_valid:<7} | {total_invalid:<7} | "
        f"{'':<23} | {'':<8} | {'':<10} | {'':<11}"
    )
    print(separator + "\n")


def main() -> None:
    """CLI entrypoint for ingestion script."""
    backend_root = Path(__file__).resolve().parent.parent
    config = IngestConfig(
        raw_data_dir=backend_root / "data" / "raw",
        db_path=backend_root / "data" / "duckdb" / "commodities.duckdb",
        table_name="onion_prices",
        sheet_name="Data Sheet",
    )

    logger.info("Starting ingestion into %s...", config.db_path)
    summaries = ingest_data(config)
    print_ingest_report(summaries, config.db_path)
    logger.info("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
