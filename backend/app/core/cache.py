"""Cache and data-handle store for data_ref and forecast_id."""

from datetime import datetime
from typing import Any, Optional
import uuid
import pandas as pd
from pydantic import BaseModel, ConfigDict


class CachedDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_ref: str
    series_id: str
    created_at: datetime
    df: Any  # pd.DataFrame
    summary: dict[str, Any]

    def to_points(self) -> list[dict[str, Any]]:
        """Convert underlying DataFrame to JSON-serializable list of points."""
        if isinstance(self.df, pd.DataFrame) and not self.df.empty:
            df_copy = self.df.copy()
            if "date" in df_copy.columns:
                df_copy["date"] = df_copy["date"].astype(str)
            return df_copy.to_dict(orient="records")
        return []


class CachedForecast(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    forecast_id: str
    target_series_id: str
    horizon_days: int
    created_at: datetime
    df: Any  # pd.DataFrame containing date, p10, p50, p90
    summary: dict[str, Any]
    risk_alerts: list[dict[str, Any]]
    backtest: Optional[dict[str, Any]] = None

    def to_points(self) -> list[dict[str, Any]]:
        if isinstance(self.df, pd.DataFrame) and not self.df.empty:
            df_copy = self.df.copy()
            if "date" in df_copy.columns:
                df_copy["date"] = df_copy["date"].astype(str)
            return df_copy.to_dict(orient="records")
        return []


class DataCache:
    """In-memory cache singleton for data references and forecasts."""

    def __init__(self) -> None:
        self._datasets: dict[str, CachedDataset] = {}
        self._forecasts: dict[str, CachedForecast] = {}

    def store_dataset(self, series_id: str, df: pd.DataFrame, summary: dict[str, Any]) -> str:
        data_ref = f"data_{uuid.uuid4().hex[:12]}"
        self._datasets[data_ref] = CachedDataset(
            data_ref=data_ref,
            series_id=series_id,
            created_at=datetime.now(),
            df=df,
            summary=summary,
        )
        return data_ref

    def get_dataset(self, data_ref: str) -> Optional[CachedDataset]:
        return self._datasets.get(data_ref)

    def store_forecast(
        self,
        target_series_id: str,
        horizon_days: int,
        df: pd.DataFrame,
        summary: dict[str, Any],
        risk_alerts: list[dict[str, Any]],
        backtest: Optional[dict[str, Any]] = None,
    ) -> str:
        forecast_id = f"fc_{uuid.uuid4().hex[:12]}"
        self._forecasts[forecast_id] = CachedForecast(
            forecast_id=forecast_id,
            target_series_id=target_series_id,
            horizon_days=horizon_days,
            created_at=datetime.now(),
            df=df,
            summary=summary,
            risk_alerts=risk_alerts,
            backtest=backtest,
        )
        return forecast_id

    def get_forecast(self, forecast_id: str) -> Optional[CachedForecast]:
        return self._forecasts.get(forecast_id)

    def clear(self) -> None:
        self._datasets.clear()
        self._forecasts.clear()


# Global cache singleton instance
global_cache = DataCache()
