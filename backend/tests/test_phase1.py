"""Phase 1 verification tests.

Validates DuckDB database ingestion, schema, and semantic layer resolution.
"""

from pathlib import Path
import duckdb
import pytest
from app.core.semantic_layer import SemanticLayer, SeriesDef, CommodityDef


@pytest.fixture
def backend_dir() -> Path:
    """Fixture providing path to backend directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def db_path(backend_dir: Path) -> Path:
    """Fixture providing path to DuckDB database."""
    return backend_dir / "data" / "duckdb" / "commodities.duckdb"


@pytest.fixture
def semantic_layer(backend_dir: Path) -> SemanticLayer:
    """Fixture providing SemanticLayer instance."""
    config_path = backend_dir / "app" / "config" / "semantic_layer.yaml"
    return SemanticLayer(config_path=config_path)


class TestDuckDBIngestion:
    """Tests verifying DuckDB ingestion results."""

    def test_database_and_table_exist(self, db_path: Path):
        """Verify the database file exists and contains onion_prices table."""
        assert db_path.exists(), f"Database file not found at {db_path}"

        con = duckdb.connect(str(db_path), read_only=True)
        try:
            tables = con.execute("SHOW TABLES").fetchall()
            table_names = [t[0] for t in tables]
            assert "onion_prices" in table_names, f"Table onion_prices not found in {table_names}"
        finally:
            con.close()

    def test_table_schema_and_date_type(self, db_path: Path):
        """Verify column types, specifically ensuring date is DATE type."""
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            columns_info = con.execute("DESCRIBE onion_prices").fetchall()
            col_dict = {col[0]: col[1].upper() for col in columns_info}

            assert "date" in col_dict, "Missing 'date' column"
            assert col_dict["date"] == "DATE", f"Expected DATE type for date column, got {col_dict['date']}"

            assert "state" in col_dict
            assert "crop" in col_dict
            assert "arrival_quantity" in col_dict
            assert "modal_price" in col_dict
            assert "retail_price" in col_dict
            assert "is_valid" in col_dict
            assert col_dict["is_valid"] == "BOOLEAN"
        finally:
            con.close()

    def test_row_counts_per_state(self, db_path: Path):
        """Verify that all 4 states have ingested records."""
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                """
                SELECT state, COUNT(*)
                FROM onion_prices
                GROUP BY state
                ORDER BY state
                """
            ).fetchall()

            state_counts = dict(rows)
            expected_states = [
                "Andhra Pradesh",
                "Karnataka",
                "Madhya Pradesh",
                "Maharashtra",
            ]

            for state in expected_states:
                assert state in state_counts, f"State {state} missing from table"
                assert state_counts[state] > 0, f"State {state} has 0 rows"
                # Typically several years of daily data (e.g. > 1000 rows each)
                assert state_counts[state] >= 1000, f"Expected >= 1000 rows for {state}, got {state_counts[state]}"
        finally:
            con.close()


class TestSemanticLayer:
    """Tests verifying semantic layer mapping and lookups."""

    def test_resolve_commodity(self, semantic_layer: SemanticLayer):
        """Verify commodity resolution by name, synonyms, and unknown handling."""
        assert semantic_layer.resolve_commodity("onion") == "onion"
        assert semantic_layer.resolve_commodity("Onion") == "onion"
        assert semantic_layer.resolve_commodity("pyaz") == "onion"
        assert semantic_layer.resolve_commodity("pyaaz") == "onion"
        assert semantic_layer.resolve_commodity("kanda") == "onion"
        assert semantic_layer.resolve_commodity("ulli") == "onion"
        assert semantic_layer.resolve_commodity("vengayam") == "onion"
        assert semantic_layer.resolve_commodity("tomato") is None
        assert semantic_layer.resolve_commodity("") is None

    def test_resolve_state(self, semantic_layer: SemanticLayer):
        """Verify state resolution with abbreviations and canonical names."""
        assert semantic_layer.resolve_state("MH") == "Maharashtra"
        assert semantic_layer.resolve_state("maha") == "Maharashtra"
        assert semantic_layer.resolve_state("maharashtra") == "Maharashtra"
        assert semantic_layer.resolve_state("AP") == "Andhra Pradesh"
        assert semantic_layer.resolve_state("andhra") == "Andhra Pradesh"
        assert semantic_layer.resolve_state("KA") == "Karnataka"
        assert semantic_layer.resolve_state("MP") == "Madhya Pradesh"
        assert semantic_layer.resolve_state("XYZ") is None

    def test_defaults(self, semantic_layer: SemanticLayer):
        """Verify default state and price type values."""
        assert semantic_layer.default_state() == "Maharashtra"
        assert semantic_layer.default_price_type() == "modal_price"

    def test_list_series_count_and_roles(self, semantic_layer: SemanticLayer):
        """Verify that all 12 series are loaded and filterable."""
        all_series = semantic_layer.list_series()
        assert len(all_series) == 12, f"Expected 12 series, found {len(all_series)}"

        targets = semantic_layer.list_series(role="target_candidate")
        # 4 states * 2 price types (modal, retail) = 8 targets
        assert len(targets) == 8, f"Expected 8 target candidates, got {len(targets)}"

        factors = semantic_layer.list_series(role="factor")
        # 4 states * 1 arrival factor = 4 factors
        assert len(factors) == 4, f"Expected 4 factors, got {len(factors)}"

    def test_get_series(self, semantic_layer: SemanticLayer):
        """Verify retrieval of specific SeriesDef by ID."""
        series_id = "onion_modal_price_maharashtra"
        s = semantic_layer.get_series(series_id)
        assert s is not None, f"Series {series_id} not found"
        assert isinstance(s, SeriesDef)
        assert s.id == series_id
        assert s.commodity == "onion"
        assert s.value_column == "modal_price"
        assert s.date_column == "date"
        assert s.filters.get("state") == "Maharashtra"
        assert s.unit == "Rs/Qtl"
        assert s.role == "target_candidate"

    def test_find_series_for(self, semantic_layer: SemanticLayer):
        """Verify helper search for commodity series by filters."""
        results = semantic_layer.find_series_for(
            commodity="pyaz",
            state="MH",
            price_type="modal_price",
        )
        assert len(results) == 1
        assert results[0].id == "onion_modal_price_maharashtra"
