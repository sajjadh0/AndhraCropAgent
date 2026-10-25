"""FastAPI request and response schemas (Section 12)."""

from typing import Any, Optional
from pydantic import BaseModel, Field
from app.core.blocks import UIBlock


class ChatRequest(BaseModel):
    session_id: str = Field(default="default_session", description="Session identifier for state persistence.")
    message: str = Field(description="User prompt or question.")


class ChatResponse(BaseModel):
    session_id: str
    trace_id: str
    blocks: list[Any] = Field(description="Ordered list of UI blocks (text, chart, risk_alert, notice, forecast_summary).")


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    model_loaded: bool
    version: str


class CatalogItem(BaseModel):
    id: str
    description: str
    commodity: str
    state: str
    unit: str
    grain: str
    role: str
    data_end_date: Optional[str] = None


class CatalogResponse(BaseModel):
    commodities: list[str]
    series: list[CatalogItem]


class ResetSessionRequest(BaseModel):
    session_id: str


class ResetSessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
