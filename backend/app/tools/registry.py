"""Tool schema registry and dependency binder for LangGraph agent.

Hard Rule #8: All tool inputs and outputs are validated with Pydantic models.
Section 8: Enums wherever values come from a fixed list (series ids, horizons, granularities).
"""

from datetime import date
from enum import Enum
from typing import Any, Callable, Literal, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.cache import DataCache, global_cache
from app.core.data_access import DataAccess
from app.core.models import ToolResult
from app.core.semantic_layer import SemanticLayer
from app.core.forecaster import ChronosForecaster
from app.core.risk import RiskChecker
from app.tools.aggregate_tool import aggregate_series
from app.tools.catalog_tool import list_available_data
from app.tools.compare_tool import compare_periods
from app.tools.forecast_tool import forecast_price
from app.tools.risk_tool import get_risk_thresholds
from app.tools.series_tool import get_series
from app.tools.stubs import search_policy_docs, web_search


class ListAvailableDataInput(BaseModel):
    """Input for listing available data catalog."""
    pass


class GetSeriesInput(BaseModel):
    """Input schema for fetching historical series."""

    series_id: str = Field(
        description="ID of the series to fetch (e.g. 'onion_modal_price_maharashtra', 'onion_retail_price_karnataka'). Do NOT use for forecasting future prices."
    )
    start_date: str = Field(
        description="Start date in YYYY-MM-DD format (e.g. '2025-01-01')."
    )
    end_date: str = Field(
        description="End date in YYYY-MM-DD format (e.g. '2026-09-30')."
    )
    granularity: Literal["daily", "weekly", "monthly"] = Field(
        default="daily",
        description="Aggregation granularity. Default is 'daily'."
    )


class ComparePeriodsInput(BaseModel):
    """Input schema for comparing two time periods."""

    series_id: str = Field(
        description="ID of the series to compare (e.g. 'onion_modal_price_maharashtra')."
    )
    period_a_start: str = Field(
        description="Start of first period in YYYY-MM-DD format (e.g. '2024-01-01')."
    )
    period_a_end: str = Field(
        description="End of first period in YYYY-MM-DD format (e.g. '2024-12-31')."
    )
    period_b_start: str = Field(
        description="Start of second period in YYYY-MM-DD format (e.g. '2025-01-01')."
    )
    period_b_end: str = Field(
        description="End of second period in YYYY-MM-DD format (e.g. '2025-12-31')."
    )
    granularity: Literal["daily", "weekly", "monthly"] = Field(
        default="daily",
        description="Granularity for alignment."
    )


class AggregateSeriesInput(BaseModel):
    """Input schema for grouped time-series aggregation."""

    series_id: str = Field(
        description="ID of the series to aggregate (e.g. 'onion_modal_price_maharashtra')."
    )
    start_date: str = Field(
        description="Start date in YYYY-MM-DD format."
    )
    end_date: str = Field(
        description="End date in YYYY-MM-DD format."
    )
    group_by: Literal["month", "year", "state"] = Field(
        default="month",
        description="Grouping dimension: 'month', 'year', or 'state'."
    )
    agg: Literal["avg", "min", "max"] = Field(
        default="avg",
        description="Aggregation function: 'avg', 'min', or 'max'."
    )


class ForecastPriceInput(BaseModel):
    """Input schema for forecasting prices."""

    target_series_id: str = Field(
        description="ID of the target price series to forecast (e.g. 'onion_modal_price_maharashtra', 'onion_retail_price_karnataka')."
    )
    horizon_days: int = Field(
        description="Number of days to forecast into the future. Allowed: 1 to 90 (common: 7, 15, 30, 90)."
    )
    factor_series_ids: Optional[list[str]] = Field(
        default=None,
        description="Optional list of factor series IDs to incorporate as covariates (e.g. ['onion_arrival_maharashtra'])."
    )
    scenario: Optional[str] = Field(
        default=None,
        description="Optional what-if scenario description."
    )


class RiskThresholdsInput(BaseModel):
    """Input schema for risk threshold inquiry."""

    commodity: str = Field(
        description="Commodity name (e.g. 'onion')."
    )


class PolicyDocsInput(BaseModel):
    """Input schema for searching policy documents."""

    query: str = Field(
        description="Policy query keywords or question."
    )


class WebSearchInput(BaseModel):
    """Input schema for web search query."""

    query: str = Field(
        description="Search query string."
    )


def build_tool_registry(
    semantic_layer: SemanticLayer,
    data_access: DataAccess,
    cache: DataCache,
    forecaster: Optional[ChronosForecaster] = None,
    risk_checker: Optional[RiskChecker] = None,
    as_of_date: Optional[date] = None,
) -> list[Any]:
    """Build and bind LangChain tool objects with injected dependencies.

    Returns list of tools bound with Pydantic validation.
    """
    if forecaster is None:
        forecaster = ChronosForecaster.get_instance()
    if risk_checker is None:
        risk_checker = RiskChecker()

    @tool(args_schema=ListAvailableDataInput)
    def tool_list_available_data() -> str:
        """Returns the full series catalog: what data exists, date ranges, and units. Use when the user asks what data you have, or when unsure which series ID to pick. Do NOT use when fetching actual prices."""
        res = list_available_data(semantic_layer, data_access)
        return res.to_llm_str()

    @tool(args_schema=GetSeriesInput)
    def tool_get_series(
        series_id: str,
        start_date: str,
        end_date: str,
        granularity: Literal["daily", "weekly", "monthly"] = "daily",
    ) -> str:
        """Fetches historical price or arrival data for a series between two dates. Returns summary stats (min, max, mean, latest) and creates a line chart. Do NOT use for forecasting future periods or comparing two periods."""
        try:
            sd = date.fromisoformat(start_date.strip())
            ed = date.fromisoformat(end_date.strip())
        except ValueError as e:
            return ToolResult(
                status="INVALID_INPUT",
                message=f"Invalid date format: {e}. Must be YYYY-MM-DD.",
            ).to_llm_str()

        res = get_series(
            series_id=series_id.strip(),
            start_date=sd,
            end_date=ed,
            granularity=granularity,
            semantic_layer=semantic_layer,
            data_access=data_access,
            cache=cache,
        )
        return res.to_llm_str()

    @tool(args_schema=ComparePeriodsInput)
    def tool_compare_periods(
        series_id: str,
        period_a_start: str,
        period_a_end: str,
        period_b_start: str,
        period_b_end: str,
        granularity: Literal["daily", "weekly", "monthly"] = "daily",
    ) -> str:
        """Compares price metrics for a series across two distinct time periods (e.g. 2024 vs 2025). Returns averages, percentage changes, and a comparison chart. Do NOT use for simple single-period history."""
        try:
            pa_s = date.fromisoformat(period_a_start.strip())
            pa_e = date.fromisoformat(period_a_end.strip())
            pb_s = date.fromisoformat(period_b_start.strip())
            pb_e = date.fromisoformat(period_b_end.strip())
        except ValueError as e:
            return ToolResult(
                status="INVALID_INPUT",
                message=f"Invalid date format: {e}. Must be YYYY-MM-DD.",
            ).to_llm_str()

        res = compare_periods(
            series_id=series_id.strip(),
            period_a_start=pa_s,
            period_a_end=pa_e,
            period_b_start=pb_s,
            period_b_end=pb_e,
            granularity=granularity,
            semantic_layer=semantic_layer,
            data_access=data_access,
            cache=cache,
        )
        return res.to_llm_str()

    @tool(args_schema=AggregateSeriesInput)
    def tool_aggregate_series(
        series_id: str,
        start_date: str,
        end_date: str,
        group_by: Literal["month", "year", "state"] = "month",
        agg: Literal["avg", "min", "max"] = "avg",
    ) -> str:
        """Computes grouped statistics (average, min, or max by month or year) and generates a bar chart. Use when the user asks for monthly averages or seasonal breakdowns. Do NOT use for daily time series or forecasts."""
        try:
            sd = date.fromisoformat(start_date.strip())
            ed = date.fromisoformat(end_date.strip())
        except ValueError as e:
            return ToolResult(
                status="INVALID_INPUT",
                message=f"Invalid date format: {e}. Must be YYYY-MM-DD.",
            ).to_llm_str()

        res = aggregate_series(
            series_id=series_id.strip(),
            start_date=sd,
            end_date=ed,
            group_by=group_by,
            agg=agg,
            semantic_layer=semantic_layer,
            data_access=data_access,
        )
        return res.to_llm_str()

    @tool(args_schema=PolicyDocsInput)
    def tool_search_policy_docs(query: str) -> str:
        """Searches policy and government documents regarding commodities. Currently a stub returning NO_DATA."""
        return search_policy_docs(query).to_llm_str()

    @tool(args_schema=WebSearchInput)
    def tool_web_search(query: str) -> str:
        """Searches the web for latest commodity news. Currently disabled for this demo."""
        return web_search(query).to_llm_str()

    @tool(args_schema=RiskThresholdsInput)
    def tool_get_risk_thresholds(commodity: str) -> str:
        """Returns the configured price risk thresholds (watch, warning, critical) for a commodity. Use when the user asks about risk levels or price thresholds."""
        res = get_risk_thresholds(commodity.strip(), risk_checker)
        return res.to_llm_str()

    @tool(args_schema=ForecastPriceInput)
    def tool_forecast_price(
        target_series_id: str,
        horizon_days: int,
        factor_series_ids: Optional[list[str]] = None,
        scenario: Optional[str] = None,
    ) -> str:
        """Forecasts future prices for 1 to 90 days (7, 15, 30, or 90 days) using Amazon Chronos-2, evaluates price risk thresholds, and checks holdout reliability. Use ONLY when asked to predict or forecast future prices. Do NOT use for past historical data."""
        res = forecast_price(
            target_series_id=target_series_id.strip(),
            horizon_days=horizon_days,
            factor_series_ids=factor_series_ids,
            scenario=scenario,
            semantic_layer=semantic_layer,
            data_access=data_access,
            forecaster=forecaster,
            risk_checker=risk_checker,
            cache=cache,
            as_of_date=as_of_date,
        )
        return res.to_llm_str()

    tools_list = [
        tool_list_available_data,
        tool_get_series,
        tool_compare_periods,
        tool_aggregate_series,
        tool_forecast_price,
        tool_get_risk_thresholds,
        tool_search_policy_docs,
        tool_web_search,
    ]

    return tools_list
