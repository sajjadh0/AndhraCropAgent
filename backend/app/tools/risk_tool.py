"""Tool 6: get_risk_thresholds.

Returns configured commodity thresholds and risk parameters.
"""

from app.core.models import ToolResult
from app.core.risk import RiskChecker


def get_risk_thresholds(
    commodity: str,
    risk_checker: RiskChecker,
) -> ToolResult:
    """Returns the configured price thresholds for a commodity."""
    cfg = risk_checker.get_thresholds(commodity)
    if not cfg:
        return ToolResult(
            status="NO_DATA",
            message=f"No risk thresholds configured for commodity '{commodity}'.",
            summary={"commodity": commodity, "reason": "no_thresholds_configured"},
        )

    levels_info = []
    for lvl in cfg.levels:
        levels_info.append(
            {
                "name": lvl.name,
                "display_name": lvl.display_name,
                "direction": lvl.direction,
                "value": lvl.value,
                "unit": cfg.unit,
                "status": lvl.status,
                "description": lvl.description,
            }
        )

    lines = [
        f"Configured Risk Thresholds for {cfg.commodity.capitalize()} ({cfg.note}):"
    ]
    for lvl in cfg.levels:
        lines.append(
            f"- {lvl.display_name}: {lvl.value:.0f} {cfg.unit} ({lvl.direction.upper()}) - {lvl.description}"
        )

    return ToolResult(
        status="OK",
        message="\n".join(lines),
        summary={
            "commodity": cfg.commodity,
            "unit": cfg.unit,
            "note": cfg.note,
            "levels": levels_info,
        },
    )
