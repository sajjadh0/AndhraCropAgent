"""State definition for LangGraph agent (Section 13)."""

from typing import Annotated, Any, Literal, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.agent.guardrails import ScopeResult
from app.core.blocks import UIBlock


class AgentState(BaseModel):
    """The graph execution state maintained across turns."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    session_id: str = "default_session"
    as_of_date: str = "2026-10-15"
    scope_result: Optional[ScopeResult] = None
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    ui_blocks: list[Any] = Field(default_factory=list)
    last_forecast_id: Optional[str] = None
    last_data_ref: Optional[str] = None
    iteration_count: int = 0
    status: Literal["OK", "REJECTED", "ERROR", "MAX_ITERATIONS"] = "OK"
    final_text: str = ""
