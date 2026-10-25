"""Settings management for the commodity price forecasting system."""

from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field


class Settings:
    """Loads settings from settings.yaml with dotted key lookup."""

    def __init__(self, config_path: Optional[Path | str] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
        self._config_path = Path(config_path)
        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config: dict[str, Any] = yaml.safe_load(f) or {}

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Lookup setting by dot notation, e.g. 'forecast.quantiles'."""
        keys = dotted_key.split(".")
        val: Any = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    @property
    def raw(self) -> dict[str, Any]:
        return self._config
