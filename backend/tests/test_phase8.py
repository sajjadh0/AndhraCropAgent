"""Tests for Phase 8: LangGraph agent execution with scripted mock LLM and routing tests."""

from datetime import date
from pathlib import Path
from typing import Any, List
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall

from app.agent.graph import create_agent_graph
from app.agent.guardrails import CANNED_REJECT_MESSAGE, CANNED_WELCOME_MESSAGE
from app.agent.state import AgentState
from app.core.cache import DataCache
from app.core.data_access import DataAccess
from app.core.forecaster import ChronosForecaster
from app.core.risk import RiskChecker
from app.core.semantic_layer import SemanticLayer


class ScriptedMockLLM:
    """Mock LLM that returns scripted tool calls or text responses."""

    def __init__(self, responses: List[AIMessage]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    def bind_tools(self, tools: Any) -> "ScriptedMockLLM":
        return self

    def invoke(self, messages: Any) -> AIMessage:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return AIMessage(content="Final answered response based on tools.")


@pytest.fixture
def agent_environment():
    backend_root = Path(__file__).resolve().parent.parent
    db_path = backend_root / "data" / "duckdb" / "commodities.duckdb"
    yaml_path = backend_root / "app" / "config" / "semantic_layer.yaml"
    thresholds_path = backend_root / "app" / "config" / "thresholds.yaml"

    semantic = SemanticLayer(yaml_path)
    data_access = DataAccess(db_path)
    cache = DataCache()
    forecaster = ChronosForecaster.get_instance()
    risk_checker = RiskChecker(thresholds_path)
    return semantic, data_access, cache, forecaster, risk_checker


class TestAgentGraphRouting:
    def test_off_topic_rejected_without_llm(self, agent_environment):
        """Off topic queries must not invoke the main LLM (Section 2, Hard Rule 7)."""
        semantic, data_access, cache, forecaster, risk_checker = agent_environment
        mock_llm = ScriptedMockLLM([])  # Will fail if invoked

        graph = create_agent_graph(
            llm=mock_llm,
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
            forecaster=forecaster,
            risk_checker=risk_checker,
        )

        res = graph.invoke(
            {"messages": [HumanMessage(content="When was Batman released?")]},
            config={"configurable": {"thread_id": "session_test_1"}},
        )

        assert res["status"] == "REJECTED"
        assert res["final_text"] == CANNED_REJECT_MESSAGE
        assert mock_llm.call_count == 0  # LLM never called!

    def test_greeting_returns_welcome(self, agent_environment):
        semantic, data_access, cache, forecaster, risk_checker = agent_environment
        mock_llm = ScriptedMockLLM([])

        graph = create_agent_graph(
            llm=mock_llm,
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
            forecaster=forecaster,
            risk_checker=risk_checker,
        )

        res = graph.invoke(
            {"messages": [HumanMessage(content="Hi")]},
            config={"configurable": {"thread_id": "session_test_2"}},
        )

        assert res["status"] == "REJECTED"
        assert res["final_text"] == CANNED_WELCOME_MESSAGE
        assert mock_llm.call_count == 0

    def test_tool_calling_flow_get_series(self, agent_environment):
        """Simulate an agent invoking get_series then answering."""
        semantic, data_access, cache, forecaster, risk_checker = agent_environment

        tool_call = {
            "name": "tool_get_series",
            "args": {
                "series_id": "onion_modal_price_maharashtra",
                "start_date": "2026-09-01",
                "end_date": "2026-10-15",
                "granularity": "daily",
            },
            "id": "call_123",
        }
        msg1 = AIMessage(content="", tool_calls=[tool_call])
        msg2 = AIMessage(content="In September-October 2026, Maharashtra onion modal prices averaged 3613.06 Rs/Qtl.")

        mock_llm = ScriptedMockLLM([msg1, msg2])

        graph = create_agent_graph(
            llm=mock_llm,
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
            forecaster=forecaster,
            risk_checker=risk_checker,
        )

        res = graph.invoke(
            {"messages": [HumanMessage(content="What were Maharashtra onion prices last month?")]},
            config={"configurable": {"thread_id": "session_test_3"}},
        )

        assert res["status"] == "OK"
        assert mock_llm.call_count == 2
        assert len(res["ui_blocks"]) >= 2  # TextBlock + ChartBlock
        assert res["last_data_ref"] is not None

    def test_forecast_tool_calling_flow(self, agent_environment):
        """Simulate an agent invoking forecast_price tool."""
        semantic, data_access, cache, forecaster, risk_checker = agent_environment

        tool_call = {
            "name": "tool_forecast_price",
            "args": {
                "target_series_id": "onion_modal_price_maharashtra",
                "horizon_days": 15,
            },
            "id": "call_fc_456",
        }
        msg1 = AIMessage(content="", tool_calls=[tool_call])
        msg2 = AIMessage(content="The projected 15-day median price is estimated with confidence bands.")

        mock_llm = ScriptedMockLLM([msg1, msg2])

        graph = create_agent_graph(
            llm=mock_llm,
            semantic_layer=semantic,
            data_access=data_access,
            cache=cache,
            forecaster=forecaster,
            risk_checker=risk_checker,
        )

        res = graph.invoke(
            {"messages": [HumanMessage(content="Forecast onion prices in Maharashtra for 15 days")]},
            config={"configurable": {"thread_id": "session_test_4"}},
        )

        assert res["status"] == "OK"
        assert res["last_forecast_id"] is not None
        block_types = [b.type for b in res["ui_blocks"]]
        assert "chart" in block_types
        assert "forecast_summary" in block_types
        assert "risk_alert" in block_types
