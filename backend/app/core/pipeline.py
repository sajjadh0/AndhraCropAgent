"""Data preparation and cleaning pipeline for forecasting.

Implements Section 9.1 and 9.2:
- History window selection based on horizon
- Sorting, numeric coercion
- Duplicate aggregation
- Continuous daily regularization
- Gap filling (linear interpolation <= 3 days, weekly fallback if > 20% filled)
- Outlier detection (IQR) and conditional capping
- Hard 'No nulls guarantee' assertion
- Validation report generation
"""

from datetime import date, timedelta
import logging
from typing import Any, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger("pipeline")


class PipelineValidationReport(BaseModel):
    """Validation report accompanying cleaned data (Section 9.2)."""

    target_series_id: str
    horizon_days: int
    requested_window_days: int
    actual_rows: int
    date_start: str
    date_end: str
    frequency: Literal["D", "W"]  # Daily or Weekly
    is_weekly_fallback: bool = False
    status: Literal["OK", "PARTIAL", "NO_DATA"] = "OK"
    warnings: list[str] = Field(default_factory=list)
    gaps_filled_count: int = 0
    filled_ratio: float = 0.0
    outliers_flagged: int = 0
    outlier_dates: list[str] = Field(default_factory=list)


class CleanedSeriesResult(BaseModel, arbitrary_types_allowed=True):
    """Result of pipeline cleaning containing ready-for-inference series and report."""

    series_df: Any  # pd.DataFrame with index=date, column='value'
    report: PipelineValidationReport


def get_history_window_days(horizon_days: int) -> Tuple[int, int]:
    """Return (recommended_history_days, min_history_days) based on Section 9.1."""
    if horizon_days <= 15:
        return 180, 90
    elif horizon_days <= 30:
        return 365, 120
    else:
        return 730, 180


def clean_series_pipeline(
    df_raw: pd.DataFrame,
    target_series_id: str,
    horizon_days: int,
    as_of_date: date,
    cap_outliers: bool = False,
    max_gap_days: int = 3,
    weekly_fallback_threshold: float = 0.20,
    max_history_days: int = 1095,
) -> CleanedSeriesResult:
    """Execute the full 8-step cleaning pipeline in Section 9.2."""
    warnings: list[str] = []
    recommended_days, min_days = get_history_window_days(horizon_days)

    # 1. Select window ending at as_of_date
    window_days = min(recommended_days, max_history_days)
    start_target = as_of_date - timedelta(days=window_days)

    if df_raw.empty:
        report = PipelineValidationReport(
            target_series_id=target_series_id,
            horizon_days=horizon_days,
            requested_window_days=window_days,
            actual_rows=0,
            date_start=str(start_target),
            date_end=str(as_of_date),
            frequency="D",
            status="NO_DATA",
            warnings=["Insufficient history: zero rows provided to pipeline."],
        )
        return CleanedSeriesResult(series_df=pd.DataFrame(), report=report)

    df = df_raw.copy()

    # 2. Sort by date and convert value to numeric
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Filter window up to as_of_date
    as_of_dt = pd.to_datetime(as_of_date)
    start_dt = pd.to_datetime(start_target)
    df = df[(df["date"] >= start_dt) & (df["date"] <= as_of_dt)]

    # Check minimum history
    actual_history_days = (as_of_dt - df["date"].min()).days if not df.empty else 0
    if actual_history_days < min_days or len(df) < (min_days * 0.4):
        report = PipelineValidationReport(
            target_series_id=target_series_id,
            horizon_days=horizon_days,
            requested_window_days=window_days,
            actual_rows=len(df),
            date_start=str(df["date"].min().date()) if not df.empty else str(start_target),
            date_end=str(as_of_date),
            frequency="D",
            status="NO_DATA",
            warnings=[
                f"Insufficient history: {actual_history_days} days available, but minimum {min_days} days required for horizon {horizon_days}."
            ],
        )
        return CleanedSeriesResult(series_df=df, report=report)

    status: Literal["OK", "PARTIAL", "NO_DATA"] = "OK"
    if actual_history_days < recommended_days:
        status = "PARTIAL"
        warnings.append(
            f"History span ({actual_history_days} days) is below recommended ({recommended_days} days) for horizon {horizon_days}."
        )

    # 3. Duplicates: if multiple rows per date, aggregate by mean
    if df["date"].duplicated().any():
        dup_count = int(df["date"].duplicated().sum())
        logger.info("Aggregating %d duplicate date entries by mean.", dup_count)
        df = df.groupby("date", as_index=False)["value"].mean()

    # 4. Regularize to a continuous daily index
    df = df.set_index("date")
    full_daily_idx = pd.date_range(start=df.index.min(), end=as_of_dt, freq="D")
    df_daily = df.reindex(full_daily_idx)

    # 5. Gaps: fill gaps <= max_gap_days (3) by linear interpolation
    missing_before = int(df_daily["value"].isna().sum())
    # Identify gaps > max_gap_days
    # limit=3 in interpolate fills at most 3 consecutive NaN
    df_interpolated = df_daily["value"].interpolate(method="linear", limit=max_gap_days)
    filled_count = missing_before - int(df_interpolated.isna().sum())
    total_expected = len(df_daily)
    filled_ratio = filled_count / max(total_expected, 1)

    is_weekly_fallback = False
    freq: Literal["D", "W"] = "D"

    # If remaining nulls exist or filled ratio > weekly_fallback_threshold (20%)
    if df_interpolated.isna().any() or filled_ratio > weekly_fallback_threshold:
        is_weekly_fallback = True
        freq = "W"
        warnings.append(
            f"Filled ratio ({filled_ratio:.1%}) exceeded threshold ({weekly_fallback_threshold:.0%}) or large gaps existed. Resampled to weekly steps."
        )
        # Resample original daily data to weekly mean
        df_resampled = df.resample("W-SUN")["value"].mean()
        # Interpolate weekly gaps up to 2 weeks
        df_clean = df_resampled.interpolate(method="linear", limit=2).bfill().ffill()
    else:
        # Bfill/ffill any boundary edges (e.g. initial 1-2 days)
        df_clean = df_interpolated.bfill().ffill()

    # 6. Outliers: detect with robust method (median / IQR)
    q25 = float(df_clean.quantile(0.25))
    q75 = float(df_clean.quantile(0.75))
    iqr = q75 - q25
    lower_bound = q25 - 2.5 * iqr
    upper_bound = q75 + 2.5 * iqr

    outlier_mask = (df_clean < lower_bound) | (df_clean > upper_bound)
    outlier_count = int(outlier_mask.sum())
    outlier_dates: list[str] = []
    if outlier_count > 0:
        outlier_dates = [d.strftime("%Y-%m-%d") for d in df_clean[outlier_mask].index]
        warnings.append(f"Flagged {outlier_count} statistical outlier(s) using IQR.")
        if cap_outliers:
            df_clean = df_clean.clip(lower=max(lower_bound, 0.0), upper=upper_bound)
            warnings.append("Outliers were capped to IQR boundaries.")

    # 7. No nulls guarantee: assert no nulls remain
    if df_clean.isna().any():
        null_count = int(df_clean.isna().sum())
        raise ValueError(
            f"Pipeline integrity failure: {null_count} null value(s) remain after interpolation and fallback."
        )

    # 8. Produce validation report
    cleaned_df = pd.DataFrame({"value": df_clean})
    cleaned_df.index.name = "date"

    report = PipelineValidationReport(
        target_series_id=target_series_id,
        horizon_days=horizon_days,
        requested_window_days=window_days,
        actual_rows=len(cleaned_df),
        date_start=cleaned_df.index.min().strftime("%Y-%m-%d"),
        date_end=cleaned_df.index.max().strftime("%Y-%m-%d"),
        frequency=freq,
        is_weekly_fallback=is_weekly_fallback,
        status=status,
        warnings=warnings,
        gaps_filled_count=filled_count,
        filled_ratio=round(filled_ratio, 4),
        outliers_flagged=outlier_count,
        outlier_dates=outlier_dates[:10],  # show first 10
    )

    return CleanedSeriesResult(series_df=cleaned_df, report=report)
