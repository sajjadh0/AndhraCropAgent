"""Tests for Phase 3: Data Pipeline (window, cleaning, gap handling, no-nulls assertion)."""

from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest
from app.core.pipeline import clean_series_pipeline, get_history_window_days


class TestPipelineHistoryWindows:
    def test_window_thresholds(self):
        rec_7, min_7 = get_history_window_days(7)
        assert rec_7 == 180 and min_7 == 90

        rec_15, min_15 = get_history_window_days(15)
        assert rec_15 == 180 and min_15 == 90

        rec_30, min_30 = get_history_window_days(30)
        assert rec_30 == 365 and min_30 == 120

        rec_90, min_90 = get_history_window_days(90)
        assert rec_90 == 730 and min_90 == 180


class TestPipelineCleaningSteps:
    def test_duplicates_handling(self):
        """Duplicates on the same date should be averaged."""
        dates = pd.date_range("2025-01-01", "2025-08-01", freq="D")
        df = pd.DataFrame({
            "date": list(dates) + [pd.Timestamp("2025-05-01")],
            "value": list(np.random.uniform(2000, 3000, len(dates))) + [4000.0],
        })
        res = clean_series_pipeline(
            df_raw=df,
            target_series_id="test_series",
            horizon_days=15,
            as_of_date=date(2025, 8, 1),
        )
        assert res.report.status == "OK"
        assert not res.series_df.index.duplicated().any()
        assert not res.series_df["value"].isna().any()

    def test_small_gaps_interpolation(self):
        """Gaps of 1-2 days should be filled by linear interpolation."""
        dates = pd.date_range("2025-01-01", "2025-08-01", freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.linspace(2000, 3000, len(dates)),
        })
        # Drop 2 consecutive days in mid-April
        df = df[~df["date"].isin([pd.Timestamp("2025-04-10"), pd.Timestamp("2025-04-11")])]

        res = clean_series_pipeline(
            df_raw=df,
            target_series_id="test_series",
            horizon_days=15,
            as_of_date=date(2025, 8, 1),
        )
        assert res.report.gaps_filled_count >= 2
        assert res.report.frequency == "D"
        assert not res.series_df["value"].isna().any()

    def test_large_gap_triggers_weekly_fallback(self):
        """Large gaps (> 3 days) or high missing ratio trigger weekly fallback."""
        dates = pd.date_range("2025-01-01", "2025-08-01", freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.linspace(2000, 3000, len(dates)),
        })
        # Remove a 10-day block in March
        drop_dates = pd.date_range("2025-03-01", "2025-03-10", freq="D")
        df = df[~df["date"].isin(drop_dates)]

        res = clean_series_pipeline(
            df_raw=df,
            target_series_id="test_series",
            horizon_days=15,
            as_of_date=date(2025, 8, 1),
        )
        assert res.report.is_weekly_fallback is True
        assert res.report.frequency == "W"
        assert not res.series_df["value"].isna().any()

    def test_outlier_detection_and_capping(self):
        """Outliers should be flagged, and capped only when cap_outliers=True."""
        dates = pd.date_range("2025-01-01", "2025-08-01", freq="D")
        values = np.random.normal(2500, 100, len(dates))
        values[50] = 12000.0  # extreme outlier

        df = pd.DataFrame({"date": dates, "value": values})

        # Default cap_outliers=False
        res_nocap = clean_series_pipeline(
            df_raw=df,
            target_series_id="test_series",
            horizon_days=15,
            as_of_date=date(2025, 8, 1),
            cap_outliers=False,
        )
        assert res_nocap.report.outliers_flagged >= 1

        # With cap_outliers=True
        res_capped = clean_series_pipeline(
            df_raw=df,
            target_series_id="test_series",
            horizon_days=15,
            as_of_date=date(2025, 8, 1),
            cap_outliers=True,
        )
        assert res_capped.series_df["value"].max() < 10000.0
        assert not res_capped.series_df["value"].isna().any()

    def test_insufficient_history_returns_no_data(self):
        """Fewer days than hard minimum returns NO_DATA."""
        dates = pd.date_range("2025-07-01", "2025-08-01", freq="D")  # only 31 days
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.uniform(2000, 3000, len(dates)),
        })
        res = clean_series_pipeline(
            df_raw=df,
            target_series_id="test_series",
            horizon_days=30,  # min required is 120
            as_of_date=date(2025, 8, 1),
        )
        assert res.report.status == "NO_DATA"
        assert "Insufficient history" in res.report.warnings[0]

    def test_no_nulls_guarantee(self):
        """Pipeline must never return a DataFrame with nulls."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        df = pd.DataFrame({
            "date": dates,
            "value": np.random.uniform(1500, 2500, len(dates)),
        })
        # Randomly punch 20 missing values
        null_indices = np.random.choice(len(df), size=20, replace=False)
        df.loc[null_indices, "value"] = np.nan

        res = clean_series_pipeline(
            df_raw=df,
            target_series_id="test_series",
            horizon_days=15,
            as_of_date=date(2024, 12, 31),
        )
        assert not res.series_df["value"].isna().any()
        assert len(res.series_df) > 0
