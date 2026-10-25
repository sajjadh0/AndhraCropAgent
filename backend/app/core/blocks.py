"""UI block models and builders for API responses.

All response blocks are pure data representations, never executable code.
"""

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field


class ChartPoint(BaseModel):
    date: str
    value: Optional[float] = None


class ChartSeries(BaseModel):
    name: str
    data_ref: Optional[str] = None
    points: Optional[list[dict[str, Any]]] = None


class Annotation(BaseModel):
    type: Literal["line", "marker"] = "line"
    label: str
    value: Any  # date string or numeric threshold
    axis: Literal["x", "y"] = "x"
    style: Literal["solid", "dashed", "dotted"] = "dashed"
    color: Optional[str] = None


class ChartBlock(BaseModel):
    type: Literal["chart"] = "chart"
    template: Literal[
        "line", "line_compare", "forecast_band", "scenario_compare", "bar", "multi_series", "table"
    ]
    title: str
    x_label: str = "Date"
    y_label: str = "Price"
    y_unit: str = "Rs/Qtl"
    series: list[ChartSeries] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)


class RiskCrossing(BaseModel):
    level: str  # watch, warning, critical
    direction: Literal["upper", "lower"]
    first_crossing_date: str
    days_away: int
    likelihood: Literal["likely", "possible"]
    threshold_value: float
    projected_value: float


class RiskAlertBlock(BaseModel):
    type: Literal["risk_alert"] = "risk_alert"
    crossings: list[RiskCrossing] = Field(default_factory=list)
    message: str = ""


class NoticeBlock(BaseModel):
    type: Literal["notice"] = "notice"
    style: Literal["popup", "inline"] = "inline"
    text: str


class ForecastSummaryBlock(BaseModel):
    type: Literal["forecast_summary"] = "forecast_summary"
    horizon_days: int
    end_value: Optional[float] = None
    pct_change: Optional[float] = None
    range_low: Optional[float] = None
    range_high: Optional[float] = None
    backtest_mape: Optional[float] = None
    backtest_mae: Optional[float] = None


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    content: str


UIBlock = Union[ChartBlock, RiskAlertBlock, NoticeBlock, ForecastSummaryBlock, TextBlock]


def build_line_chart(
    title: str,
    series_name: str,
    data_ref: str,
    y_label: str = "Price",
    y_unit: str = "Rs/Qtl",
    as_of_date: Optional[str] = None,
) -> ChartBlock:
    annotations: list[Annotation] = []
    if as_of_date:
        annotations.append(Annotation(type="marker", label="Latest Data", value=as_of_date, axis="x"))
    return ChartBlock(
        template="line",
        title=title,
        y_label=y_label,
        y_unit=y_unit,
        series=[ChartSeries(name=series_name, data_ref=data_ref)],
        annotations=annotations,
    )


def build_line_compare_chart(
    title: str,
    series_a_name: str,
    data_ref_a: str,
    series_b_name: str,
    data_ref_b: str,
    y_label: str = "Price",
    y_unit: str = "Rs/Qtl",
) -> ChartBlock:
    return ChartBlock(
        template="line_compare",
        title=title,
        y_label=y_label,
        y_unit=y_unit,
        series=[
            ChartSeries(name=series_a_name, data_ref=data_ref_a),
            ChartSeries(name=series_b_name, data_ref=data_ref_b),
        ],
    )


def build_bar_chart(
    title: str,
    series_name: str,
    points: list[dict[str, Any]],
    x_label: str = "Category",
    y_label: str = "Value",
    y_unit: str = "Rs/Qtl",
) -> ChartBlock:
    return ChartBlock(
        template="bar",
        title=title,
        x_label=x_label,
        y_label=y_label,
        y_unit=y_unit,
        series=[ChartSeries(name=series_name, points=points)],
    )


def build_forecast_band_chart(
    title: str,
    history_ref: str,
    forecast_ref: str,
    as_of_date: str,
    y_label: str = "Price",
    y_unit: str = "Rs/Qtl",
    threshold_annotations: Optional[list[Annotation]] = None,
) -> ChartBlock:
    annotations = [Annotation(type="marker", label="Today", value=as_of_date, axis="x")]
    if threshold_annotations:
        annotations.extend(threshold_annotations)
    return ChartBlock(
        template="forecast_band",
        title=title,
        y_label=y_label,
        y_unit=y_unit,
        series=[
            ChartSeries(name="Historical", data_ref=history_ref),
            ChartSeries(name="Forecast (p10/p50/p90)", data_ref=forecast_ref),
        ],
        annotations=annotations,
    )


def build_notice(text: str, style: Literal["popup", "inline"] = "inline") -> NoticeBlock:
    return NoticeBlock(style=style, text=text)


def build_stale_data_notice(as_of_date: str) -> NoticeBlock:
    return build_notice(f"Latest data available is dated {as_of_date}.", style="inline")


def build_no_data_notice(web_search_enabled: bool = False) -> NoticeBlock:
    if web_search_enabled:
        return build_notice("Searching the web for information...", style="inline")
    return build_notice(
        "No data about this is available, and web search is turned off for this demo.",
        style="popup",
    )
