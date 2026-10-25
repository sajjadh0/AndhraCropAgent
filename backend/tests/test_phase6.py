"""Tests for Phase 6: Factor stitching with covariates, baseline-vs-factor, and forecast_price tool."""

from datetime import date
from pathlib import Path
import pytest

from app.core.cache import DataCache
from app.core.data_access import DataAccess
from app.core.forecaster import ChronosForecaster
from app.core.risk import RiskChecker
from app.core.semantic_layer import SemanticLayer
from app.tools.forecast_tool import forecast_price


@pytest.fixture
def service_setup():
    backend_root = Path(__file__).resolve().parent.parent
    db_path = backend_root / "data" / "duckdb" / "commodities.duckdb"
    yaml_path = backend_root / "app" / "config" / "semantic_layer.yaml"
    thresholds_path = backend_root / "app" / "config" / "thresholds.yaml"

    semantic = SemanticLayer(yaml_path)
    data_access = DataAccess(db_path)
    cache = DataCache()
    forecaster = ChronosForecaster.get_instance()
    risk_checker = RiskChecker(thresholds_path)
    as_of = date(2026, 10, 15)

    return semantic, data_access, forecaster, risk_checker, cache, as_of


class TestForecastPriceTool:
    def test_forecast_price_baseline_7_days(self, service_setup):
        semantic, data_access, forecaster, risk_checker, cache, as_of = service_setup
        res = forecast_price(
            target_series_id="onion_modal_price_maharashtra",
            horizon_days=7,
            semantic_layer=semantic,
            data_access=data_access,
            forecaster=forecaster,
            risk_checker=risk_checker,
            cache=cache,
            as_of_date=as_of,
        )
        assert res.status == "OK"
        assert res.forecast_id is not None
        assert res.data_ref is not None
        assert res.summary["horizon_days"] == 7
        assert "expected_value_end" in res.summary

        # Check UI blocks: chart, summary, risk
        block_types = [b.type for b in res.ui_blocks]
        assert "chart" in block_types
        assert "forecast_summary" in block_types
        assert "risk_alert" in block_types

    def test_forecast_price_with_arrival_factor(self, service_setup):
        """End-to-end forecast incorporating arrival quantities as covariate factor."""
        semantic, data_access, forecaster, risk_checker, cache, as_of = service_setup
        res = forecast_price(
            target_series_id="onion_modal_price_maharashtra",
            horizon_days=15,
            factor_series_ids=["onion_arrival_maharashtra"],
            semantic_layer=semantic,
            data_access=data_access,
            forecaster=forecaster,
            risk_checker=risk_checker,
            cache=cache,
            as_of_date=as_of,
        )
        assert res.status == "OK"
        assert res.forecast_id is not None
        # Check that chart contains both forecast and baseline
        chart_block = next(b for b in res.ui_blocks if b.type == "chart")
        series_names = [s.name for s in chart_block.series]
        assert "Baseline (without factors)" in series_names

    def test_forecast_price_invalid_horizon_over_90(self, service_setup):
        semantic, data_access, forecaster, risk_checker, cache, as_of = service_setup
        res = forecast_price(
            target_series_id="onion_modal_price_maharashtra",
            horizon_days=200,
            semantic_layer=semantic,
            data_access=data_access,
            forecaster=forecaster,
            risk_checker=risk_checker,
            cache=cache,
            as_of_date=as_of,
        )
        assert res.status == "INVALID_INPUT"
        assert "must be between 1 and 90" in res.message

    def test_forecast_price_unknown_factor(self, service_setup):
        semantic, data_access, forecaster, risk_checker, cache, as_of = service_setup
        res = forecast_price(
            target_series_id="onion_modal_price_maharashtra",
            horizon_days=15,
            factor_series_ids=["unknown_fuel_factor"],
            semantic_layer=semantic,
            data_access=data_access,
            forecaster=forecaster,
            risk_checker=risk_checker,
            cache=cache,
            as_of_date=as_of,
        )
        assert res.status == "NO_DATA"
        assert "unknown_fuel_factor" in res.message
