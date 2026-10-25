"""Tests for Phase 4: Chronos Forecaster (horizons 7, 15, 30, 90, quantiles ordering, backtesting)."""

import numpy as np
import pandas as pd
import pytest
from app.core.forecaster import ChronosForecaster


@pytest.fixture
def sample_history():
    dates = pd.date_range("2024-01-01", "2025-10-15", freq="D")
    # Generate realistic seasonal prices
    t = np.arange(len(dates))
    base = 2200 + 400 * np.sin(2 * np.pi * t / 365) + np.cumsum(np.random.normal(0, 5, len(dates)))
    df = pd.DataFrame({"value": np.maximum(base, 500.0)}, index=dates)
    df.index.name = "date"
    return df


class TestChronosForecaster:
    @pytest.mark.parametrize("horizon", [7, 15, 30, 90])
    def test_forecast_horizons_and_quantiles(self, sample_history, horizon):
        forecaster = ChronosForecaster.get_instance()
        res = forecaster.forecast(
            df_cleaned=sample_history,
            horizon_days=horizon,
            quantiles=[0.1, 0.5, 0.9],
        )

        assert res.horizon_days == horizon
        assert len(res.forecast_df) == horizon
        assert "p10" in res.forecast_df.columns
        assert "p50" in res.forecast_df.columns
        assert "p90" in res.forecast_df.columns

        # Quantile ordering assertion: p10 <= p50 <= p90
        p10 = res.forecast_df["p10"].values
        p50 = res.forecast_df["p50"].values
        p90 = res.forecast_df["p90"].values

        assert np.all(p10 <= p50 + 1e-5), "p10 must be <= p50"
        assert np.all(p50 <= p90 + 1e-5), "p50 must be <= p90"
        assert np.all(p10 >= 0), "Prices cannot be negative"

        if horizon >= 60:
            assert len(res.caveats) > 0
            assert "Uncertainty widens" in res.caveats[0]

    def test_backtest_metrics(self, sample_history):
        forecaster = ChronosForecaster.get_instance()
        res = forecaster.forecast(
            df_cleaned=sample_history,
            horizon_days=15,
        )
        assert res.backtest is not None
        assert res.backtest.mape is not None
        assert res.backtest.mae is not None
        assert res.backtest.mape >= 0.0
