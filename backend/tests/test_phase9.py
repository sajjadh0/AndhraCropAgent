"""Tests for Phase 9: FastAPI endpoints and API response contract.

Tests all 8 endpoints and contract requirements specified in Section 12:
1. GET /api/v1/health -> HealthResponse
2. GET /api/v1/catalog -> CatalogResponse (12 series, commodities)
3. POST /api/v1/chat -> ChatResponse (off-topic canned rejection)
4. POST /api/v1/chat -> ChatResponse (greeting canned response)
5. POST /api/v1/chat -> ChatResponse (in-scope forecast with chart and metrics)
6. GET /api/v1/data/{data_ref} -> 404 for unknown, points list for stored ref
7. GET /api/v1/forecast/{forecast_id} -> 404 for unknown
8. POST /api/v1/session/reset -> ResetSessionResponse
"""

from fastapi import HTTPException
import pandas as pd
import pytest

from app.api.routes import (
    get_catalog,
    get_data_ref,
    get_forecast_id,
    get_health,
    post_chat,
    post_reset_session,
)
from app.api.schemas import ChatRequest, ResetSessionRequest
from app.main import app_state


class TestAPIEndpoints:
    def test_health_endpoint(self):
        resp = get_health()
        assert resp.status == "healthy"
        assert resp.database_connected is True
        assert resp.model_loaded is True
        assert resp.version == "0.1.0"

    def test_catalog_endpoint(self):
        resp = get_catalog()
        assert "onion" in resp.commodities
        assert len(resp.series) == 12
        series_ids = [s.id for s in resp.series]
        assert "onion_modal_price_maharashtra" in series_ids
        assert "onion_arrival_maharashtra" in series_ids

    def test_chat_off_topic_rejection(self):
        req = ChatRequest(
            session_id="test_sess_offtopic",
            message="When was Batman released?",
        )
        resp = post_chat(req)
        assert resp.session_id == "test_sess_offtopic"
        assert resp.trace_id.startswith("trace_")
        assert len(resp.blocks) >= 1
        assert resp.blocks[0].type == "text"
        assert "only help with commodity prices" in resp.blocks[0].content

    def test_chat_greeting(self):
        req = ChatRequest(
            session_id="test_sess_greeting",
            message="Hello",
        )
        resp = post_chat(req)
        assert len(resp.blocks) >= 1
        assert "Commodity Price Analysis assistant" in resp.blocks[0].content

    def test_chat_in_scope_forecast(self):
        req = ChatRequest(
            session_id="test_sess_fc",
            message="Forecast onion prices in Maharashtra for 15 days",
        )
        resp = post_chat(req)
        assert len(resp.blocks) >= 1
        block_types = [b.type for b in resp.blocks]
        assert "text" in block_types
        assert "chart" in block_types or "forecast_summary" in block_types

    def test_data_ref_lookup(self):
        # 1. Unknown data ref -> 404 HTTPException
        with pytest.raises(HTTPException) as exc_info:
            get_data_ref("nonexistent_ref_999")
        assert exc_info.value.status_code == 404

        # 2. Store dummy dataset in cache and fetch
        df = pd.DataFrame([{"date": "2026-10-10", "value": 2500.0}])
        ref = app_state.cache.store_dataset("onion_modal_price_maharashtra", df, {"mean": 2500.0})

        resp = get_data_ref(ref)
        assert resp["data_ref"] == ref
        assert resp["series_id"] == "onion_modal_price_maharashtra"
        assert len(resp["points"]) == 1
        assert resp["points"][0]["value"] == 2500.0

    def test_forecast_id_lookup(self):
        # Unknown forecast_id -> 404 HTTPException
        with pytest.raises(HTTPException) as exc_info:
            get_forecast_id("nonexistent_fc_999")
        assert exc_info.value.status_code == 404

    def test_session_reset(self):
        req = ResetSessionRequest(session_id="test_sess_to_reset")
        resp = post_reset_session(req)
        assert resp.session_id == "test_sess_to_reset"
        assert resp.status == "OK"
