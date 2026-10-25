"""Tool 1: list_available_data.

Returns the catalog of available commodities, series, date ranges, and units.
"""

from typing import Any
from app.core.data_access import DataAccess
from app.core.models import ToolResult
from app.core.semantic_layer import SemanticLayer


def list_available_data(
    semantic_layer: SemanticLayer,
    data_access: DataAccess,
) -> ToolResult:
    """Returns the series catalog: what data exists, date ranges, units."""
    catalog: list[dict[str, Any]] = []
    for s in semantic_layer.list_series():
        end_d = data_access.data_end_date(s)
        start_d = data_access.data_start_date(s)
        catalog.append(
            {
                "id": s.id,
                "description": s.description,
                "commodity": s.commodity,
                "state": s.filters.get("state", "All"),
                "unit": s.unit,
                "grain": s.grain,
                "role": s.role,
                "date_range": f"{start_d} to {end_d}" if start_d and end_d else "Unknown",
            }
        )

    summary = {
        "series_count": len(catalog),
        "commodities": semantic_layer.all_commodity_keys,
        "series_list": [c["id"] for c in catalog],
    }

    formatted_lines = [
        f"Available Data Catalog ({len(catalog)} series registered):"
    ]
    for c in catalog:
        formatted_lines.append(
            f"- {c['id']}: {c['description']} | State: {c['state']} | Range: {c['date_range']} | Unit: {c['unit']}"
        )

    return ToolResult(
        status="OK",
        message="\n".join(formatted_lines),
        summary=summary,
    )
