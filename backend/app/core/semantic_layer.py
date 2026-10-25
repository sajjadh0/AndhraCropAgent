"""Semantic layer for commodity forecasting.

Provides domain models and lookup functions to map user queries and entities
to canonical database series, tables, and dimensions.
"""

from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field


class CommodityDef(BaseModel):
    """Commodity definition containing canonical name, synonyms, and defaults."""

    canonical_name: str
    synonyms: list[str] = Field(default_factory=list)
    unit: str
    default_state: str


class SeriesDef(BaseModel):
    """Time-series definition mapping to database table, column, and filters."""

    id: str
    description: str
    commodity: str
    source_table: str
    value_column: str
    date_column: str
    filters: dict[str, str] = Field(default_factory=dict)
    grain: str  # daily, weekly, monthly
    unit: str
    role: str  # target_candidate or factor


class SemanticLayer:
    """Loads and queries the semantic layer configuration."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        """Initialize semantic layer from YAML config file.

        Args:
            config_path: Optional path to semantic_layer.yaml. If None,
                defaults to backend/app/config/semantic_layer.yaml.
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "config" / "semantic_layer.yaml"
        self._config_path = Path(config_path)
        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config: dict[str, Any] = yaml.safe_load(f) or {}

        self._commodities: dict[str, CommodityDef] = {}
        self._series: dict[str, SeriesDef] = {}
        self._dimensions: dict[str, Any] = {}
        self._local_defaults: dict[str, Any] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse raw YAML dictionary into typed definitions."""
        # Parse commodities
        for key, val in self._config.get("commodities", {}).items():
            self._commodities[key] = CommodityDef(**val)

        # Parse series
        for key, val in self._config.get("series", {}).items():
            self._series[key] = SeriesDef(id=key, **val)

        # Parse dimensions and local defaults
        self._dimensions = self._config.get("dimensions", {})
        self._local_defaults = self._config.get("local_defaults", {})

    def resolve_commodity(self, text: str) -> Optional[str]:
        """Resolve user text to canonical commodity key.

        Args:
            text: Commodity name or synonym (e.g. 'onion', 'pyaz', 'kanda').

        Returns:
            Canonical commodity key (e.g. 'onion') or None if not recognized.
        """
        if not text:
            return None
        text_lower = text.strip().lower()
        for key, commodity in self._commodities.items():
            if text_lower == key.lower() or text_lower == commodity.canonical_name.lower():
                return key
            if text_lower in [s.lower() for s in commodity.synonyms]:
                return key
        return None

    def resolve_series(self, text_or_id: str) -> Optional[SeriesDef]:
        """Resolve a series ID or description text to a SeriesDef.

        Args:
            text_or_id: Exact series ID or partial text matching description.

        Returns:
            Matching SeriesDef or None.
        """
        if not text_or_id:
            return None
        # Direct ID match
        if text_or_id in self._series:
            return self._series[text_or_id]

        # Case-insensitive direct ID match
        text_lower = text_or_id.strip().lower()
        for sid, sdef in self._series.items():
            if sid.lower() == text_lower:
                return sdef

        # Match on description
        for sid, sdef in self._series.items():
            if text_lower in sdef.description.lower():
                return sdef
        return None

    def list_series(
        self,
        role: str | None = None,
        commodity: str | None = None,
    ) -> list[SeriesDef]:
        """List all series, optionally filtered by role and/or commodity.

        Args:
            role: Optional role filter ('target_candidate' or 'factor').
            commodity: Optional commodity filter key.

        Returns:
            List of matching SeriesDef objects.
        """
        result = list(self._series.values())
        if role:
            result = [s for s in result if s.role == role]
        if commodity:
            commodity_key = self.resolve_commodity(commodity) or commodity
            result = [s for s in result if s.commodity == commodity_key]
        return result

    def get_series(self, series_id: str) -> Optional[SeriesDef]:
        """Get a series by exact ID.

        Args:
            series_id: Series identifier key.

        Returns:
            SeriesDef or None if not found.
        """
        return self._series.get(series_id)

    def get_commodity(self, key: str) -> Optional[CommodityDef]:
        """Get a commodity definition by key.

        Args:
            key: Commodity key (e.g. 'onion').

        Returns:
            CommodityDef or None if not found.
        """
        return self._commodities.get(key)

    def resolve_state(self, text: str) -> Optional[str]:
        """Resolve user text to a canonical state name.

        Args:
            text: State name or synonym/abbreviation (e.g. 'MH', 'AP', 'Karnataka').

        Returns:
            Canonical state name (e.g. 'Maharashtra') or None if not recognized.
        """
        if not text:
            return None
        text_lower = text.strip().lower()
        states = self._dimensions.get("states", {}).get("values", [])
        for state in states:
            name = state.get("name", "")
            if text_lower == name.lower():
                return name
            synonyms = state.get("synonyms", [])
            if text_lower in [s.lower() for s in synonyms]:
                return name
        return None

    def default_state(self) -> str:
        """Get the default state from dimension configuration."""
        return self._dimensions.get("states", {}).get("default", "Maharashtra")

    def resolve_price_type(self, text: str) -> Optional[str]:
        """Resolve price type text to canonical column name.

        Args:
            text: Price type description or synonym (e.g. 'mandi', 'retail').

        Returns:
            Canonical column name ('modal_price' or 'retail_price') or None.
        """
        if not text:
            return None
        text_lower = text.strip().lower()
        price_types = self._dimensions.get("price_types", {}).get("values", [])
        for pt in price_types:
            name = pt.get("name", "")
            if text_lower == name.lower():
                return name
            synonyms = pt.get("synonyms", [])
            if text_lower in [s.lower() for s in synonyms]:
                return name
        return None

    def default_price_type(self) -> str:
        """Get the default price type from dimension configuration."""
        return self._dimensions.get("price_types", {}).get("default", "modal_price")

    def find_series_for(
        self,
        commodity: str,
        state: str | None = None,
        price_type: str | None = None,
        role: str | None = None,
    ) -> list[SeriesDef]:
        """Find series matching commodity, state, price_type, and role.

        Args:
            commodity: Commodity name or synonym.
            state: Optional state name or synonym.
            price_type: Optional price type or column name.
            role: Optional role ('target_candidate' or 'factor').

        Returns:
            List of matching SeriesDef objects.
        """
        commodity_key = self.resolve_commodity(commodity)
        if not commodity_key:
            return []

        results = [s for s in self._series.values() if s.commodity == commodity_key]

        if state:
            resolved_state = self.resolve_state(state)
            if resolved_state:
                results = [s for s in results if s.filters.get("state") == resolved_state]
            else:
                return []

        if price_type:
            resolved_pt = self.resolve_price_type(price_type) or price_type
            results = [s for s in results if s.value_column == resolved_pt]

        if role:
            results = [s for s in results if s.role == role]

        return results

    @property
    def all_series_ids(self) -> list[str]:
        """List of all registered series IDs."""
        return list(self._series.keys())

    @property
    def all_commodity_keys(self) -> list[str]:
        """List of all registered commodity keys."""
        return list(self._commodities.keys())
