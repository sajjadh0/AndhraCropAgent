"""Tool 3: compare_periods.

Compares a series between two distinct time periods (e.g. 2023 vs 2024),
computes comparative statistics, and builds a line_compare chart block.
"""

from datetime import date
from typing import Literal, Optional
from app.core.blocks import build_line_compare_chart, build_no_data_notice
from app.core.cache import DataCache
from app.core.data_access import DataAccess
from app.core.models import ToolResult
from app.core.semantic_layer import SemanticLayer


def compare_periods(
    series_id: str,
    period_a_start: date,
    period_a_end: date,
    period_b_start: date,
    period_b_end: date,
    granularity: Literal["daily", "weekly", "monthly"] = "daily",
    semantic_layer: Optional[SemanticLayer] = None,
    data_access: Optional[DataAccess] = None,
    cache: Optional[DataCache] = None,
) -> ToolResult:
    """Compare two periods for a series, e.g., last year vs this year."""
    if semantic_layer is None or data_access is None or cache is None:
        raise ValueError("Dependencies must be injected.")

    # Validation
    if period_a_start > period_a_end:
        return ToolResult(
            status="INVALID_INPUT",
            message=f"period_a_start ({period_a_start}) must be <= period_a_end ({period_a_end}).",
            summary={"error": "inverted_period_a"},
        )
    if period_b_start > period_b_end:
        return ToolResult(
            status="INVALID_INPUT",
            message=f"period_b_start ({period_b_start}) must be <= period_b_end ({period_b_end}).",
            summary={"error": "inverted_period_b"},
        )

    series_def = semantic_layer.resolve_series(series_id)
    if not series_def:
        return ToolResult(
            status="NO_DATA",
            message=f"Unknown series '{series_id}'.",
            summary={"reason": "unknown_series"},
            ui_blocks=[build_no_data_notice(web_search_enabled=False)],
        )

    df_a = data_access.fetch_series(series_def, period_a_start, period_a_end)
    df_b = data_access.fetch_series(series_def, period_b_start, period_b_end)

    if df_a.empty or df_b.empty:
        missing = []
        if df_a.empty:
            missing.append(f"Period A ({period_a_start} to {period_a_end})")
        if df_b.empty:
            missing.append(f"Period B ({period_b_start} to {period_b_end})")
        return ToolResult(
            status="NO_DATA",
            message=f"Insufficient data: zero rows in {' and '.join(missing)}.",
            summary={"reason": "empty_period_data"},
            ui_blocks=[build_no_data_notice(web_search_enabled=False)],
        )

    val_a = df_a["value"].dropna()
    val_b = df_b["value"].dropna()

    mean_a = float(val_a.mean())
    mean_b = float(val_b.mean())
    pct_diff = float(((mean_b - mean_a) / mean_a) * 100) if mean_a != 0 else 0.0

    summary_a = {
        "period": f"{period_a_start} to {period_a_end}",
        "count": len(df_a),
        "mean": round(mean_a, 2),
        "min": round(float(val_a.min()), 2),
        "max": round(float(val_a.max()), 2),
    }
    summary_b = {
        "period": f"{period_b_start} to {period_b_end}",
        "count": len(df_b),
        "mean": round(mean_b, 2),
        "min": round(float(val_b.min()), 2),
        "max": round(float(val_b.max()), 2),
    }

    ref_a = cache.store_dataset(f"{series_def.id}_period_a", df_a, summary_a)
    ref_b = cache.store_dataset(f"{series_def.id}_period_b", df_b, summary_b)

    chart = build_line_compare_chart(
        title=f"{series_def.description}: Period Comparison",
        series_a_name=f"Period A ({period_a_start} to {period_a_end})",
        data_ref_a=ref_a,
        series_b_name=f"Period B ({period_b_start} to {period_b_end})",
        data_ref_b=ref_b,
        y_label="Price",
        y_unit=series_def.unit,
    )

    summary = {
        "series_id": series_def.id,
        "period_a": summary_a,
        "period_b": summary_b,
        "mean_difference": round(mean_b - mean_a, 2),
        "pct_difference": round(pct_diff, 2),
    }

    message = (
        f"Compared {series_def.description} between Period A ({period_a_start} to {period_a_end}) "
        f"and Period B ({period_b_start} to {period_b_end}). "
        f"Period A average was {mean_a:.2f} {series_def.unit}, while Period B average was "
        f"{mean_b:.2f} {series_def.unit} ({pct_diff:+.2f}% change)."
    )

    return ToolResult(
        status="OK",
        message=message,
        data_ref=ref_b,
        summary=summary,
        ui_blocks=[chart],
    )
