"""Inspect all data files in data/raw/ and print a detailed report."""

import pandas as pd
from pathlib import Path
import sys

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def inspect_file(filepath: Path) -> None:
    print(f"\n{'='*80}")
    print(f"FILE: {filepath.name}")
    print(f"Size: {filepath.stat().st_size:,} bytes")
    print(f"{'='*80}")

    try:
        # Read all sheets for Excel files
        if filepath.suffix in (".xlsx", ".xls"):
            xls = pd.ExcelFile(filepath)
            sheets = xls.sheet_names
            print(f"Sheets: {sheets}")
            for sheet in sheets:
                print(f"\n--- Sheet: '{sheet}' ---")
                df = pd.read_excel(filepath, sheet_name=sheet)
                _print_df_info(df)
        elif filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
            _print_df_info(df)
        else:
            print(f"  Skipping unsupported format: {filepath.suffix}")
    except Exception as e:
        print(f"  ERROR reading file: {e}")


def _print_df_info(df: pd.DataFrame) -> None:
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Columns and dtypes:")
    for col in df.columns:
        print(f"    {col!r:40s} dtype={str(df[col].dtype):15s} nulls={df[col].isna().sum():>6d}/{len(df)}")

    print(f"\n  First 5 rows:")
    print(df.head().to_string(index=False))

    print(f"\n  Last 3 rows:")
    print(df.tail(3).to_string(index=False))

    # Try to detect date columns and show range
    for col in df.columns:
        col_lower = str(col).lower()
        if "date" in col_lower or "dt" in col_lower or "time" in col_lower:
            try:
                dates = pd.to_datetime(df[col], errors="coerce")
                valid = dates.dropna()
                if len(valid) > 0:
                    print(f"\n  Date column '{col}': {valid.min()} → {valid.max()} ({len(valid)} valid dates)")
            except Exception:
                pass

    # Show unique values for string columns with < 20 unique values
    print(f"\n  Categorical column unique values:")
    for col in df.columns:
        if df[col].dtype == "object":
            uniques = df[col].dropna().unique()
            if len(uniques) <= 30:
                print(f"    {col!r}: {list(uniques)}")
            else:
                print(f"    {col!r}: {len(uniques)} unique values (showing first 10): {list(uniques[:10])}")

    # Numeric column stats
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"\n  Numeric column stats:")
        print(df[num_cols].describe().round(2).to_string())

    # Check for negative values in numeric columns
    for col in num_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"\n  WARNING: Column '{col}' has {neg_count} negative values!")


def main():
    files = sorted(RAW_DIR.glob("*"))
    if not files:
        print("No files found in", RAW_DIR)
        sys.exit(1)

    print(f"Found {len(files)} files in {RAW_DIR}:")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    for f in files:
        inspect_file(f)


if __name__ == "__main__":
    main()
