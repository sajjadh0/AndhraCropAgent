"""LangGraph orchestration graph implementing Section 13.

Graph topology:
input_guard -> [pass -> agent | fail -> reject -> END]
agent -> [has_tools -> tools -> agent | no_tools -> output_guard -> assemble -> END]
"""

from datetime import date
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agent.guardrails import (
    CANNED_REJECT_MESSAGE,
    CANNED_WELCOME_MESSAGE,
    classify_scope,
    run_deterministic_input_guard,
    run_output_guard,
    session_limiter,
)
from app.agent.state import AgentState
from app.core.blocks import TextBlock
from app.core.cache import DataCache, global_cache
from app.core.data_access import DataAccess
from app.core.forecaster import ChronosForecaster
from app.core.risk import RiskChecker
from app.core.semantic_layer import SemanticLayer
from app.tools.aggregate_tool import aggregate_series
from app.tools.catalog_tool import list_available_data
from app.tools.compare_tool import compare_periods
from app.tools.forecast_tool import forecast_price
from app.tools.registry import build_tool_registry
from app.tools.risk_tool import get_risk_thresholds
from app.tools.series_tool import get_series
from app.tools.stubs import search_policy_docs, web_search

logger = logging.getLogger("agent.graph")


def load_system_prompt(as_of_date: str, semantic_layer: SemanticLayer) -> str:
    """Load and format system prompt template."""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    series_lines = []
    for s in semantic_layer.list_series():
        series_lines.append(f"- ID: {s.id} | {s.description} ({s.unit}, {s.grain})")
    catalog_str = "\n".join(series_lines)

    return template.format(
        as_of_date=as_of_date,
        series_catalog=catalog_str,
    )


def create_agent_graph(
    llm: Any,
    semantic_layer: SemanticLayer,
    data_access: DataAccess,
    cache: Optional[DataCache] = None,
    forecaster: Optional[ChronosForecaster] = None,
    risk_checker: Optional[RiskChecker] = None,
    checkpointer: Optional[Any] = None,
    as_of_date: str = "2026-10-15",
    max_iterations: int = 6,
) -> Any:
    """Build LangGraph workflow with strict guardrails and tool execution."""
    if cache is None:
        cache = global_cache
    if forecaster is None:
        forecaster = ChronosForecaster.get_instance()
    if risk_checker is None:
        risk_checker = RiskChecker()
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Tools definition
    tools_list = build_tool_registry(
        semantic_layer=semantic_layer,
        data_access=data_access,
        cache=cache,
        forecaster=forecaster,
        risk_checker=risk_checker,
        as_of_date=date.fromisoformat(as_of_date),
    )
    tools_by_name = {t.name: t for t in tools_list}

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools_list) if hasattr(llm, "bind_tools") else llm

    system_prompt_str = load_system_prompt(as_of_date, semantic_layer)

    # 1. Input Guard Node
    def node_input_guard(state: AgentState) -> dict[str, Any]:
        last_user_msg = ""
        for m in reversed(state.messages):
            if isinstance(m, HumanMessage):
                last_user_msg = str(m.content)
                break

        # Layer 6: Abuse rate limit check
        if not session_limiter.is_allowed(state.session_id):
            return {
                "status": "REJECTED",
                "final_text": "Rate limit exceeded. Please wait a moment before sending more requests.",
                "ui_blocks": [TextBlock(content="Rate limit exceeded. Please wait a moment.")],
            }

        # Layer 1: Deterministic check
        passed, rej_msg = run_deterministic_input_guard(last_user_msg)
        if not passed:
            return {
                "status": "REJECTED",
                "final_text": rej_msg or CANNED_REJECT_MESSAGE,
                "ui_blocks": [TextBlock(content=rej_msg or CANNED_REJECT_MESSAGE)],
            }

        # Layer 2: Scope classification
        scope_res = classify_scope(last_user_msg)
        if not scope_res.in_scope:
            return {
                "status": "REJECTED",
                "scope_result": scope_res,
                "final_text": scope_res.rejection_message or CANNED_REJECT_MESSAGE,
                "ui_blocks": [TextBlock(content=scope_res.rejection_message or CANNED_REJECT_MESSAGE)],
            }

        if scope_res.category == "greeting":
            return {
                "status": "REJECTED",
                "scope_result": scope_res,
                "final_text": CANNED_WELCOME_MESSAGE,
                "ui_blocks": [TextBlock(content=CANNED_WELCOME_MESSAGE)],
            }

        return {
            "status": "OK",
            "scope_result": scope_res,
            "as_of_date": as_of_date,
        }

    # 2. Agent Node (calls LLM)
    def node_agent(state: AgentState) -> dict[str, Any]:
        current_iter = state.iteration_count + 1
        if current_iter > max_iterations:
            logger.warning("Session %s reached max iterations limit.", state.session_id)
            return {
                "iteration_count": current_iter,
                "status": "MAX_ITERATIONS",
                "final_text": "I reached the maximum reasoning steps for this query. Please refine your question.",
            }

        prompt_messages = [SystemMessage(content=system_prompt_str)] + state.messages
        ai_msg = llm_with_tools.invoke(prompt_messages)
        return {
            "messages": [ai_msg],
            "iteration_count": current_iter,
        }

    # 3. Tool Execution Node
    def node_tools(state: AgentState) -> dict[str, Any]:
        last_ai_msg = state.messages[-1]
        tool_messages = []
        accumulated_blocks = list(state.ui_blocks)
        collected_summaries = list(state.tool_results)
        last_data_ref = state.last_data_ref
        last_forecast_id = state.last_forecast_id

        if hasattr(last_ai_msg, "tool_calls"):
            for tc in last_ai_msg.tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args", {})
                t_id = tc.get("id", "call_default")

                logger.info("Executing tool: %s with args: %s", t_name, t_args)
                tool_fn = tools_by_name.get(t_name)

                start_t = time.time()
                # Run underlying core implementation to get ToolResult envelope with ui_blocks
                tool_result_obj = _execute_tool_core(
                    t_name=t_name,
                    t_args=t_args,
                    semantic_layer=semantic_layer,
                    data_access=data_access,
                    cache=cache,
                    forecaster=forecaster,
                    risk_checker=risk_checker,
                    as_of_date=as_of_date,
                )
                duration = time.time() - start_t
                logger.info("Tool %s finished in %.2fs with status %s", t_name, duration, tool_result_obj.status)

                # Collect blocks and summaries
                if tool_result_obj.ui_blocks:
                    accumulated_blocks.extend(tool_result_obj.ui_blocks)
                if tool_result_obj.summary:
                    collected_summaries.append(tool_result_obj.summary)
                if tool_result_obj.data_ref:
                    last_data_ref = tool_result_obj.data_ref
                if tool_result_obj.forecast_id:
                    last_forecast_id = tool_result_obj.forecast_id

                tool_messages.append(
                    ToolMessage(
                        tool_call_id=t_id,
                        content=tool_result_obj.to_llm_str(),
                    )
                )

        return {
            "messages": tool_messages,
            "ui_blocks": accumulated_blocks,
            "tool_results": collected_summaries,
            "last_data_ref": last_data_ref,
            "last_forecast_id": last_forecast_id,
        }

    # 4. Output Guard Node
    def node_output_guard(state: AgentState) -> dict[str, Any]:
        last_msg = state.messages[-1] if state.messages else None
        text_content = str(last_msg.content) if last_msg else ""

        passed, sanitized_text = run_output_guard(
            llm_output=text_content,
            tool_summaries=state.tool_results,
        )
        return {
            "final_text": sanitized_text,
        }

    # 5. Assemble Node
    def node_assemble(state: AgentState) -> dict[str, Any]:
        all_blocks = []
        if state.final_text:
            all_blocks.append(TextBlock(content=state.final_text))
        all_blocks.extend(state.ui_blocks)
        return {
            "ui_blocks": all_blocks,
        }

    # 6. Reject Node
    def node_reject(state: AgentState) -> dict[str, Any]:
        return {
            "ui_blocks": [TextBlock(content=state.final_text)],
        }

    # Conditional Routers
    def route_after_input_guard(state: AgentState) -> Literal["agent", "reject"]:
        if state.status == "REJECTED":
            return "reject"
        return "agent"

    def route_after_agent(state: AgentState) -> Literal["tools", "output_guard", "assemble"]:
        if state.status == "MAX_ITERATIONS":
            return "assemble"
        last_msg = state.messages[-1]
        if hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0:
            return "tools"
        return "output_guard"

    # Build Graph
    workflow = StateGraph(AgentState)

    workflow.add_node("input_guard", node_input_guard)
    workflow.add_node("agent", node_agent)
    workflow.add_node("tools", node_tools)
    workflow.add_node("output_guard", node_output_guard)
    workflow.add_node("assemble", node_assemble)
    workflow.add_node("reject", node_reject)

    workflow.set_entry_point("input_guard")

    workflow.add_conditional_edges("input_guard", route_after_input_guard)
    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_edge("tools", "agent")
    workflow.add_edge("output_guard", "assemble")
    workflow.add_edge("assemble", END)
    workflow.add_edge("reject", END)

    return workflow.compile(checkpointer=checkpointer)


def _execute_tool_core(
    t_name: str,
    t_args: dict[str, Any],
    semantic_layer: SemanticLayer,
    data_access: DataAccess,
    cache: DataCache,
    forecaster: ChronosForecaster,
    risk_checker: RiskChecker,
    as_of_date: str,
) -> Any:
    """Helper to run tool core implementation directly to harvest ToolResult with UI blocks."""
    as_of_dt = date.fromisoformat(as_of_date)

    if t_name == "tool_list_available_data":
        return list_available_data(semantic_layer, data_access)

    elif t_name == "tool_get_series":
        try:
            return get_series(
                series_id=t_args["series_id"],
                start_date=date.fromisoformat(t_args["start_date"]),
                end_date=date.fromisoformat(t_args["end_date"]),
                granularity=t_args.get("granularity", "daily"),
                semantic_layer=semantic_layer,
                data_access=data_access,
                cache=cache,
            )
        except Exception as e:
            return _make_error_tool_result(str(e))

    elif t_name == "tool_compare_periods":
        try:
            return compare_periods(
                series_id=t_args["series_id"],
                period_a_start=date.fromisoformat(t_args["period_a_start"]),
                period_a_end=date.fromisoformat(t_args["period_a_end"]),
                period_b_start=date.fromisoformat(t_args["period_b_start"]),
                period_b_end=date.fromisoformat(t_args["period_b_end"]),
                granularity=t_args.get("granularity", "daily"),
                semantic_layer=semantic_layer,
                data_access=data_access,
                cache=cache,
            )
        except Exception as e:
            return _make_error_tool_result(str(e))

    elif t_name == "tool_aggregate_series":
        try:
            return aggregate_series(
                series_id=t_args["series_id"],
                start_date=date.fromisoformat(t_args["start_date"]),
                end_date=date.fromisoformat(t_args["end_date"]),
                group_by=t_args.get("group_by", "month"),
                agg=t_args.get("agg", "avg"),
                semantic_layer=semantic_layer,
                data_access=data_access,
            )
        except Exception as e:
            return _make_error_tool_result(str(e))

    elif t_name == "tool_forecast_price":
        try:
            return forecast_price(
                target_series_id=t_args["target_series_id"],
                horizon_days=int(t_args["horizon_days"]),
                factor_series_ids=t_args.get("factor_series_ids"),
                scenario=t_args.get("scenario"),
                semantic_layer=semantic_layer,
                data_access=data_access,
                forecaster=forecaster,
                risk_checker=risk_checker,
                cache=cache,
                as_of_date=as_of_dt,
            )
        except Exception as e:
            return _make_error_tool_result(str(e))

    elif t_name == "tool_get_risk_thresholds":
        return get_risk_thresholds(t_args.get("commodity", "onion"), risk_checker)

    elif t_name == "tool_search_policy_docs":
        return search_policy_docs(t_args.get("query", ""))

    elif t_name == "tool_web_search":
        return web_search(t_args.get("query", ""))

    return _make_error_tool_result(f"Unknown tool name: {t_name}")


def _make_error_tool_result(msg: str):
    from app.core.models import ToolResult
    return ToolResult(status="ERROR", message=msg, summary={"error": msg})
