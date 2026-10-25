"""Tests for Phase 5: Risk thresholds and evaluation."""

from datetime import date
from pathlib import Path
import pandas as pd
import pytest
from app.core.risk import RiskChecker
from app.tools.risk_tool import get_risk_thresholds


@pytest.fixture
def risk_checker():
    yaml_path = Path(__file__).resolve().parent.parent / "app" / "config" / "thresholds.yaml"
    return RiskChecker(yaml_path)


class TestRiskChecker:
    def test_get_thresholds_onion(self, risk_checker):
        cfg = risk_checker.get_thresholds("onion")
        assert cfg is not None
        assert cfg.commodity == "onion"
        assert len(cfg.levels) >= 3
        level_names = [lvl.name for lvl in cfg.levels]
        assert "warning_high" in level_names
        assert "critical_high" in level_names

    def test_no_crossing(self, risk_checker):
        """Forecast within safe bounds should not trigger any crossing."""
        dates = pd.date_range("2026-10-16", periods=7, freq="D")
        forecast_df = pd.DataFrame({
            "date": [str(d.date()) for d in dates],
            "p10": [1800.0] * 7,
            "p50": [2000.0] * 7,
            "p90": [2200.0] * 7,
        })
        alert, annotations = risk_checker.evaluate_forecast(
            forecast_df=forecast_df,
            commodity="onion",
            as_of_date=date(2026, 10, 15),
        )
        assert len(alert.crossings) == 0
        assert "No price risk thresholds crossed" in alert.message
        assert len(annotations) >= 3

    def test_likely_crossing_median(self, risk_checker):
        """When median crosses upper threshold, likelihood must be 'likely'."""
        dates = pd.date_range("2026-10-16", periods=15, freq="D")
        # Median crosses warning_high (3200) on day 5
        p50 = [2500.0] * 5 + [3300.0] * 10
        forecast_df = pd.DataFrame({
            "date": [str(d.date()) for d in dates],
            "p10": [2000.0] * 15,
            "p50": p50,
            "p90": [3800.0] * 15,
        })
        alert, _ = risk_checker.evaluate_forecast(
            forecast_df=forecast_df,
            commodity="onion",
            as_of_date=date(2026, 10, 15),
        )
        assert len(alert.crossings) > 0
        warning_cross = [c for c in alert.crossings if "Warning" in c.level]
        assert len(warning_cross) == 1
        assert warning_cross[0].likelihood == "likely"
        assert warning_cross[0].days_away == 6

    def test_possible_crossing_only_p90(self, risk_checker):
        """When only outer band (p90) crosses, likelihood must be 'possible'."""
        dates = pd.date_range("2026-10-16", periods=15, freq="D")
        # Median is safe (2800 < 3200), but p90 crosses warning_high (3200)
        forecast_df = pd.DataFrame({
            "date": [str(d.date()) for d in dates],
            "p10": [2000.0] * 15,
            "p50": [2800.0] * 15,
            "p90": [3400.0] * 15,
        })
        alert, _ = risk_checker.evaluate_forecast(
            forecast_df=forecast_df,
            commodity="onion",
            as_of_date=date(2026, 10, 15),
        )
        warning_cross = [c for c in alert.crossings if "Warning" in c.level]
        assert len(warning_cross) == 1
        assert warning_cross[0].likelihood == "possible"

    def test_tool_get_risk_thresholds(self, risk_checker):
        res = get_risk_thresholds("onion", risk_checker)
        assert res.status == "OK"
        assert "levels" in res.summary
        assert "auto-derived" in res.message
