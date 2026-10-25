"""Unit and contract tests for Phase 2: Data Access, Blocks, and Tools 1-4."""

from datetime import date
from pathlib import Path
import pytest
from app.core.blocks import (
    ChartBlock,
    NoticeBlock,
    build_bar_chart,
    build_line_chart,
    build_no_data_notice,
)
from app.core.cache import DataCache
from app.core.data_access import DataAccess
from app.core.models import ToolResult
from app.core.semantic_layer import SemanticLayer
from app.tools.aggregate_tool import aggregate_series
from app.tools.catalog_tool import list_available_data
from app.tools.compare_tool import compare_periods
from app.tools.series_tool import get_series


@pytest.fixture
def test_setup():
    backend_root = Path(__file__).resolve().parent.parent
    db_path = backend_root / "data" / "duckdb" / "commodities.duckdb"
    yaml_path = backend_root / "app" / "config" / "semantic_layer.yaml"

    semantic = SemanticLayer(yaml_path)
    data_access = DataAccess(db_path)
    cache = DataCache()
    return semantic, data_access, cache


class TestDataAccess:
    def test_data_end_date(self, test_setup):
        semantic, data_access, _ = test_setup
        series_def = semantic.get_series("onion_modal_price_maharashtra")
        assert series_def is not None
        end_d = data_access.data_end_date(series_def)
        assert end_d == date(2026, 10, 15)

    def test_fetch_series_shape(self, test_setup):
        semantic, data_access, _ = test_setup
        series_def = semantic.get_series("onion_modal_price_maharashtra")
        df = data_access.fetch_series(series_def, date(2024, 1, 1), date(2024, 1, 31))
        assert not df.empty
        assert "date" in df.columns
        assert "value" in df.columns
        assert len(df) >= 20  # Mandi trading days in a month

    def test_row_count(self, test_setup):
        semantic, data_access, _ = test_setup
        series_def = semantic.get_series("onion_modal_price_karnataka")
        count = data_access.row_count(series_def, date(2024, 1, 1), date(2024, 12, 31))
        assert count > 200

    def test_fetch_aggregated_monthly(self, test_setup):
        semantic, data_access, _ = test_setup
        series_def = semantic.get_series("onion_modal_price_mp")
        df = data_access.fetch_aggregated(
            series_def, date(2024, 1, 1), date(2024, 6, 30), group_by="month", agg="avg"
        )
        assert not df.empty
        assert "group_key" in df.columns
        assert "value" in df.columns
        assert len(df) == 6  # 6 months


class TestBlocksAndModels:
    def test_tool_result_envelope(self):
        res = ToolResult(
            status="OK",
            message="Data loaded successfully.",
            data_ref="data_test123",
            summary={"mean": 2450.5, "count": 100},
            ui_blocks=[build_line_chart("Price", "MH", "data_test123")],
        )
        assert res.status == "OK"
        assert res.data_ref == "data_test123"
        llm_text = res.to_llm_str()
        assert "2450.50" in llm_text
        assert "data_test123" in llm_text

    def test_no_data_notice_popup(self):
        notice = build_no_data_notice(web_search_enabled=False)
        assert notice.type == "notice"
        assert notice.style == "popup"
        assert "web search is turned off" in notice.text


class TestTools:
    def test_catalog_tool(self, test_setup):
        semantic, data_access, _ = test_setup
        res = list_available_data(semantic, data_access)
        assert res.status == "OK"
        assert res.summary["series_count"] == 12

    def test_get_series_ok(self, test_setup):
        semantic, data_access, cache = test_setup
        res = get_series(
            series_id="onion_modal_price_maharashtra",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            granularity="daily",
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
        )
        assert res.status == "OK"
        assert res.data_ref is not None
        assert "mean" in res.summary
        assert len(res.ui_blocks) == 1
        assert res.ui_blocks[0].type == "chart"

    def test_get_series_invalid_dates(self, test_setup):
        semantic, data_access, cache = test_setup
        res = get_series(
            series_id="onion_modal_price_maharashtra",
            start_date=date(2024, 6, 1),
            end_date=date(2024, 1, 1),
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
        )
        assert res.status == "INVALID_INPUT"

    def test_get_series_unknown_series(self, test_setup):
        semantic, data_access, cache = test_setup
        res = get_series(
            series_id="tomato_modal_price",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 1),
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
        )
        assert res.status == "NO_DATA"
        assert res.ui_blocks[0].style == "popup"

    def test_compare_periods_ok(self, test_setup):
        semantic, data_access, cache = test_setup
        res = compare_periods(
            series_id="onion_modal_price_maharashtra",
            period_a_start=date(2023, 1, 1),
            period_a_end=date(2023, 3, 31),
            period_b_start=date(2024, 1, 1),
            period_b_end=date(2024, 3, 31),
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
        )
        assert res.status == "OK"
        assert "mean_difference" in res.summary
        assert "pct_difference" in res.summary
        assert len(res.ui_blocks) == 1
        assert res.ui_blocks[0].template == "line_compare"

    def test_aggregate_series_ok(self, test_setup):
        semantic, data_access, _ = test_setup
        res = aggregate_series(
            series_id="onion_modal_price_karnataka",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            group_by="month",
            agg="avg",
            semantic_layer=semantic,
            data_access=data_access,
        )
        assert res.status == "OK"
        assert res.summary["group_count"] == 12
        assert len(res.ui_blocks) == 1
        assert res.ui_blocks[0].template == "bar"

    def test_cache_retrieval(self, test_setup):
        semantic, data_access, cache = test_setup
        res = get_series(
            series_id="onion_modal_price_maharashtra",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 15),
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
        )
        dataset = cache.get_dataset(res.data_ref)
        assert dataset is not None
        points = dataset.to_points()
        assert len(points) > 0
        assert "date" in points[0]
        assert "value" in points[0]
