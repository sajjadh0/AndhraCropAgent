"""Tool 2: get_series.

Fetches historical series points, computes summary stats, builds a line chart block,
and stores the raw points under a data_ref handle.
"""

from datetime import date
from typing import Literal, Optional
import pandas as pd
from app.core.blocks import build_line_chart, build_no_data_notice, build_stale_data_notice
from app.core.cache import DataCache
from app.core.data_access import DataAccess
from app.core.models import ToolResult
from app.core.semantic_layer import SemanticLayer


def get_series(
    series_id: str,
    start_date: date,
    end_date: date,
    granularity: Literal["daily", "weekly", "monthly"] = "daily",
    semantic_layer: Optional[SemanticLayer] = None,
    data_access: Optional[DataAccess] = None,
    cache: Optional[DataCache] = None,
) -> ToolResult:
    """Fetches a historical series, returns summary stats and line chart."""
    if semantic_layer is None or data_access is None or cache is None:
        raise ValueError("Dependencies must be injected.")

    # 1. Validation: date ordering
    if start_date > end_date:
        return ToolResult(
            status="INVALID_INPUT",
            message=f"start_date ({start_date}) must be earlier than or equal to end_date ({end_date}).",
            summary={"error": "inverted_dates"},
        )

    # 2. Resolve series
    series_def = semantic_layer.resolve_series(series_id)
    if not series_def:
        return ToolResult(
            status="NO_DATA",
            message=f"Series '{series_id}' is not recognized in the semantic catalog.",
            summary={"reason": "unknown_series", "series_id": series_id},
            ui_blocks=[build_no_data_notice(web_search_enabled=False)],
        )

    # 3. Fetch data
    df = data_access.fetch_series(series_def, start_date, end_date)
    if df.empty:
        return ToolResult(
            status="NO_DATA",
            message=f"No data found for {series_def.description} between {start_date} and {end_date}.",
            summary={"reason": "zero_rows", "series_id": series_id},
            ui_blocks=[build_no_data_notice(web_search_enabled=False)],
        )

    # 4. Coverage check (70% rule for daily grain)
    total_days = (end_date - start_date).days + 1
    actual_rows = len(df)
    coverage_ratio = actual_rows / max(total_days, 1)

    status: Literal["OK", "PARTIAL"] = "OK"
    notice_blocks = []
    if series_def.grain == "daily" and coverage_ratio < 0.70:
        status = "PARTIAL"
        notice_blocks.append(
            build_no_data_notice(web_search_enabled=False)
            if actual_rows < 5
            else build_stale_data_notice(str(df["date"].iloc[-1]))
        )

    # 5. Granularity aggregation if needed
    if granularity in ("weekly", "monthly"):
        df["date_dt"] = pd.to_datetime(df["date"])
        freq = "W" if granularity == "weekly" else "MS"
        df_resampled = (
            df.set_index("date_dt")
            .resample(freq)["value"]
            .mean()
            .dropna()
            .reset_index()
        )
        df_resampled["date"] = df_resampled["date_dt"].dt.strftime("%Y-%m-%d")
        df = df_resampled[["date", "value"]]

    # 6. Compute summary stats
    values = df["value"].dropna()
    earliest_val = float(values.iloc[0])
    latest_val = float(values.iloc[-1])
    min_val = float(values.min())
    max_val = float(values.max())
    mean_val = float(values.mean())
    pct_change = float(((latest_val - earliest_val) / earliest_val) * 100) if earliest_val != 0 else 0.0

    summary = {
        "series_id": series_def.id,
        "description": series_def.description,
        "unit": series_def.unit,
        "start_date": str(df["date"].iloc[0]),
        "end_date": str(df["date"].iloc[-1]),
        "count": len(df),
        "min": round(min_val, 2),
        "max": round(max_val, 2),
        "mean": round(mean_val, 2),
        "latest": round(latest_val, 2),
        "pct_change": round(pct_change, 2),
    }

    # 7. Store dataset in cache
    data_ref = cache.store_dataset(series_def.id, df, summary)

    # 8. Build line chart block
    chart_block = build_line_chart(
        title=f"{series_def.description} ({granularity.capitalize()})",
        series_name=series_def.filters.get("state", "Price"),
        data_ref=data_ref,
        y_label="Modal Price" if "modal" in series_def.id else "Price",
        y_unit=series_def.unit,
        as_of_date=str(df["date"].iloc[-1]),
    )

    ui_blocks = [chart_block]
    if status == "PARTIAL":
        ui_blocks.extend(notice_blocks)

    return ToolResult(
        status=status,
        message=(
            f"Retrieved {len(df)} points for {series_def.description}. "
            f"Latest value is {latest_val:.2f} {series_def.unit} on {df['date'].iloc[-1]}, "
            f"with an average of {mean_val:.2f} {series_def.unit} (change: {pct_change:+.2f}%)."
        ),
        data_ref=data_ref,
        summary=summary,
        ui_blocks=ui_blocks,
    )
