"""Risk checker and threshold evaluator.

Implements Section 10:
- Compares median (p50) and quantile bands (p10/p90) against configured thresholds.
- Classifies likelihood: "likely" (median crosses) vs "possible" (only outer band crosses).
- Generates chart annotations for threshold reference lines.
- Builds RiskAlertBlock for frontend rendering.
"""

from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
import yaml

from app.core.blocks import Annotation, RiskAlertBlock, RiskCrossing


class ThresholdLevel(BaseModel):
    name: str
    display_name: str
    direction: Literal["upper", "lower"]
    value: float
    status: str = "PLACEHOLDER_AUTO_DERIVED"
    description: str = ""


class CommodityThresholds(BaseModel):
    commodity: str
    unit: str
    note: str
    levels: list[ThresholdLevel]


class RiskChecker:
    """Evaluates forecasts against commodity thresholds."""

    def __init__(self, thresholds_path: Optional[Path | str] = None) -> None:
        if thresholds_path is None:
            thresholds_path = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"
        self._path = Path(thresholds_path)
        with open(self._path, "r", encoding="utf-8") as f:
            self._data: dict[str, Any] = yaml.safe_load(f) or {}

    def get_thresholds(self, commodity: str) -> Optional[CommodityThresholds]:
        """Fetch configured threshold definition for a commodity."""
        commodity_key = commodity.lower().strip()
        commodities = self._data.get("commodities", {})
        if commodity_key not in commodities:
            return None
        cfg = commodities[commodity_key]
        return CommodityThresholds(
            commodity=commodity_key,
            unit=cfg.get("unit", "Rs/Qtl"),
            note=cfg.get("note", "auto-derived, replace with real values"),
            levels=[ThresholdLevel(**lvl) for lvl in cfg.get("levels", [])],
        )

    def evaluate_forecast(
        self,
        forecast_df: pd.DataFrame,
        commodity: str,
        as_of_date: date,
    ) -> tuple[RiskAlertBlock, list[Annotation]]:
        """Evaluate a forecast DataFrame against commodity risk thresholds.

        Priority rule:
        - If median path (p50) crosses the threshold: "likely" (first date median crossed).
        - If only outer band (p90 for upper, p10 for lower) crosses: "possible" (first date band crossed).
        """
        thresholds = self.get_thresholds(commodity)
        if not thresholds or forecast_df.empty:
            return (
                RiskAlertBlock(
                    crossings=[],
                    message="No thresholds configured for this commodity.",
                ),
                [],
            )

        crossings: list[RiskCrossing] = []
        chart_annotations: list[Annotation] = []

        level_colors = {
            "watch_high": "#E6A23C",
            "warning_high": "#F56C6C",
            "critical_high": "#C00000",
            "critical_low": "#409EFF",
        }

        as_of_dt = pd.to_datetime(as_of_date).date()

        for lvl in thresholds.levels:
            chart_annotations.append(
                Annotation(
                    type="line",
                    label=f"{lvl.display_name} ({lvl.value:.0f} {thresholds.unit})",
                    value=lvl.value,
                    axis="y",
                    style="dashed",
                    color=level_colors.get(lvl.name, "#909399"),
                )
            )

            median_crossed = False
            first_median_crossing: Optional[RiskCrossing] = None

            band_crossed = False
            first_band_crossing: Optional[RiskCrossing] = None

            for _, row in forecast_df.iterrows():
                row_date = pd.to_datetime(row["date"]).date()
                days_away = (row_date - as_of_dt).days
                p10_val = float(row["p10"])
                p50_val = float(row["p50"])
                p90_val = float(row["p90"])

                if lvl.direction == "upper":
                    if p50_val >= lvl.value and not median_crossed:
                        median_crossed = True
                        first_median_crossing = RiskCrossing(
                            level=lvl.display_name,
                            direction="upper",
                            first_crossing_date=str(row_date),
                            days_away=max(days_away, 0),
                            likelihood="likely",
                            threshold_value=lvl.value,
                            projected_value=round(p50_val, 2),
                        )
                    if p90_val >= lvl.value and not band_crossed:
                        band_crossed = True
                        first_band_crossing = RiskCrossing(
                            level=lvl.display_name,
                            direction="upper",
                            first_crossing_date=str(row_date),
                            days_away=max(days_away, 0),
                            likelihood="possible",
                            threshold_value=lvl.value,
                            projected_value=round(p90_val, 2),
                        )
                elif lvl.direction == "lower":
                    if p50_val <= lvl.value and not median_crossed:
                        median_crossed = True
                        first_median_crossing = RiskCrossing(
                            level=lvl.display_name,
                            direction="lower",
                            first_crossing_date=str(row_date),
                            days_away=max(days_away, 0),
                            likelihood="likely",
                            threshold_value=lvl.value,
                            projected_value=round(p50_val, 2),
                        )
                    if p10_val <= lvl.value and not band_crossed:
                        band_crossed = True
                        first_band_crossing = RiskCrossing(
                            level=lvl.display_name,
                            direction="lower",
                            first_crossing_date=str(row_date),
                            days_away=max(days_away, 0),
                            likelihood="possible",
                            threshold_value=lvl.value,
                            projected_value=round(p10_val, 2),
                        )

            # Assign crossing: median takes precedence
            if median_crossed and first_median_crossing:
                crossings.append(first_median_crossing)
            elif band_crossed and first_band_crossing:
                crossings.append(first_band_crossing)

        # Build message
        if not crossings:
            message = "No price risk thresholds crossed during this forecast horizon."
        else:
            alert_lines = [
                f"Price Risk Alerts ({len(crossings)} threshold(s) crossed, {thresholds.note}):"
            ]
            for c in crossings:
                alert_lines.append(
                    f"- {c.level} ({c.threshold_value:.0f} {thresholds.unit}): {c.likelihood.capitalize()} crossing on {c.first_crossing_date} (in {c.days_away} days; projected: {c.projected_value:.2f})."
                )
            message = "\n".join(alert_lines)

        return RiskAlertBlock(crossings=crossings, message=message), chart_annotations
