"""Tool 4: aggregate_series.

Aggregates a series grouped by month, year, or state, computes grouped statistics,
and generates a bar chart block with inline points.
"""

from datetime import date
from typing import Literal, Optional
from app.core.blocks import build_bar_chart, build_no_data_notice
from app.core.data_access import DataAccess
from app.core.models import ToolResult
from app.core.semantic_layer import SemanticLayer


def aggregate_series(
    series_id: str,
    start_date: date,
    end_date: date,
    group_by: Literal["month", "year", "state"] = "month",
    agg: Literal["avg", "min", "max"] = "avg",
    semantic_layer: Optional[SemanticLayer] = None,
    data_access: Optional[DataAccess] = None,
) -> ToolResult:
    """Grouped statistics and bar chart for a time series."""
    if semantic_layer is None or data_access is None:
        raise ValueError("Dependencies must be injected.")

    if start_date > end_date:
        return ToolResult(
            status="INVALID_INPUT",
            message=f"start_date ({start_date}) must be <= end_date ({end_date}).",
            summary={"error": "inverted_dates"},
        )

    series_def = semantic_layer.resolve_series(series_id)
    if not series_def:
        return ToolResult(
            status="NO_DATA",
            message=f"Unknown series '{series_id}'.",
            summary={"reason": "unknown_series"},
            ui_blocks=[build_no_data_notice(web_search_enabled=False)],
        )

    try:
        df = data_access.fetch_aggregated(
            series_def, start_date, end_date, group_by=group_by, agg=agg
        )
    except Exception as e:
        return ToolResult(
            status="ERROR",
            message=f"Aggregation query failed: {str(e)}",
            summary={"error": str(e)},
        )

    if df.empty:
        return ToolResult(
            status="NO_DATA",
            message=f"No data available to aggregate for {series_def.description} between {start_date} and {end_date}.",
            summary={"reason": "zero_rows"},
            ui_blocks=[build_no_data_notice(web_search_enabled=False)],
        )

    # Convert to chart points: [{"group": key, "value": val}]
    points = []
    for _, row in df.iterrows():
        points.append(
            {
                "date": str(row["group_key"]),
                "value": float(row["value"]) if row["value"] is not None else 0.0,
            }
        )

    values = df["value"].dropna()
    summary = {
        "series_id": series_def.id,
        "group_by": group_by,
        "agg_function": agg,
        "group_count": len(df),
        "min_group_value": round(float(values.min()), 2),
        "max_group_value": round(float(values.max()), 2),
        "mean_across_groups": round(float(values.mean()), 2),
    }

    chart = build_bar_chart(
        title=f"{series_def.description} ({agg.upper()} by {group_by.capitalize()})",
        series_name=f"{agg.upper()} Value",
        points=points,
        x_label=group_by.capitalize(),
        y_label=f"{agg.upper()} Price",
        y_unit=series_def.unit,
    )

    message = (
        f"Aggregated {series_def.description} by {group_by} ({agg.upper()}) "
        f"from {start_date} to {end_date}. Found {len(df)} groups, "
        f"ranging from {summary['min_group_value']} to {summary['max_group_value']} {series_def.unit}."
    )

    return ToolResult(
        status="OK",
        message=message,
        summary=summary,
        ui_blocks=[chart],
    )
