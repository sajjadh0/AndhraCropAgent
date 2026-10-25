"""Stub tools for future phases (policy docs search and web search)."""

from app.core.blocks import build_no_data_notice, build_notice
from app.core.models import ToolResult


def search_policy_docs(query: str) -> ToolResult:
    """Stub tool: searches policy documents. Returns NO_DATA for demo."""
    return ToolResult(
        status="NO_DATA",
        message="No policy documents loaded in the system.",
        summary={"query": query, "reason": "no_policy_docs"},
        ui_blocks=[build_notice("No policy documents loaded in the system.", style="inline")],
    )


def web_search(query: str) -> ToolResult:
    """Stub tool: web search. Returns WEB_DISABLED for demo."""
    return ToolResult(
        status="WEB_DISABLED",
        message="Web search is disabled for this demo.",
        summary={"query": query, "reason": "web_search_disabled"},
        ui_blocks=[build_no_data_notice(web_search_enabled=False)],
    )
