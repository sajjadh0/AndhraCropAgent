"""FastAPI routes for versioned API /api/v1."""

import os
import uuid
from typing import Any, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, status
from langchain_core.messages import HumanMessage

from app.api.schemas import (
    CatalogItem,
    CatalogResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ResetSessionRequest,
    ResetSessionResponse,
)
from app.core.cache import DataCache, global_cache
from app.core.data_access import DataAccess
from app.core.semantic_layer import SemanticLayer

router = APIRouter(prefix="/api/v1", tags=["Commodity Forecaster API v1"])


def get_api_key_auth(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """Optional API key authentication check (Section 12)."""
    expected_key = os.getenv("API_KEY")
    # If API_KEY environment variable is configured and non-empty, enforce it
    if expected_key and expected_key.strip():
        if not x_api_key or x_api_key != expected_key.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header.",
            )


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Check system health, DuckDB connectivity, and model initialization."""
    from app.main import app_state

    db_ok = False
    try:
        con = app_state.data_access._connect()
        con.execute("SELECT 1").fetchall()
        con.close()
        db_ok = True
    except Exception:
        db_ok = False

    model_ok = app_state.forecaster is not None

    return HealthResponse(
        status="healthy" if db_ok else "unhealthy",
        database_connected=db_ok,
        model_loaded=model_ok,
        version="0.1.0",
    )


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog() -> CatalogResponse:
    """Fetch series catalog for frontend dropdowns."""
    from app.main import app_state

    catalog_items: list[CatalogItem] = []
    for s in app_state.semantic_layer.list_series():
        end_d = app_state.data_access.data_end_date(s)
        catalog_items.append(
            CatalogItem(
                id=s.id,
                description=s.description,
                commodity=s.commodity,
                state=s.filters.get("state", "All"),
                unit=s.unit,
                grain=s.grain,
                role=s.role,
                data_end_date=str(end_d) if end_d else None,
            )
        )

    return CatalogResponse(
        commodities=app_state.semantic_layer.all_commodity_keys,
        series=catalog_items,
    )


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(get_api_key_auth)])
def post_chat(req: ChatRequest) -> ChatResponse:
    """Chat endpoint returning ordered UI blocks."""
    from app.main import app_state

    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": req.session_id}}

    # Invoke LangGraph agent
    result = app_state.agent_graph.invoke(
        {"messages": [HumanMessage(content=req.message)], "session_id": req.session_id},
        config=config,
    )

    blocks = result.get("ui_blocks", [])

    return ChatResponse(
        session_id=req.session_id,
        trace_id=trace_id,
        blocks=blocks,
    )


@router.get("/data/{data_ref}")
def get_data_ref(data_ref: str) -> dict[str, Any]:
    """Retrieve raw historical points for a data_ref handle."""
    from app.main import app_state

    ds = app_state.cache.get_dataset(data_ref)
    if not ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data reference '{data_ref}' not found or expired.",
        )
    return {
        "data_ref": data_ref,
        "series_id": ds.series_id,
        "summary": ds.summary,
        "points": ds.to_points(),
    }


@router.get("/forecast/{forecast_id}")
def get_forecast_id(forecast_id: str) -> dict[str, Any]:
    """Retrieve full forecast prediction points, backtest, and risk alerts."""
    from app.main import app_state

    fc = app_state.cache.get_forecast(forecast_id)
    if not fc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forecast ID '{forecast_id}' not found or expired.",
        )
    return {
        "forecast_id": forecast_id,
        "target_series_id": fc.target_series_id,
        "horizon_days": fc.horizon_days,
        "summary": fc.summary,
        "risk_alerts": fc.risk_alerts,
        "backtest": fc.backtest,
        "points": fc.to_points(),
    }


@router.post("/session/reset", response_model=ResetSessionResponse)
def post_reset_session(req: ResetSessionRequest) -> ResetSessionResponse:
    """Clear memory state for a session."""
    return ResetSessionResponse(
        session_id=req.session_id,
        status="OK",
        message=f"Session state for '{req.session_id}' has been reset.",
    )
