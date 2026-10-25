"""Core models including the common tool result envelope."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from app.core.blocks import UIBlock


class ToolResult(BaseModel):
    """Common envelope returned by every tool (Section 7)."""

    status: Literal["OK", "PARTIAL", "NO_DATA", "INVALID_INPUT", "WEB_DISABLED", "ERROR"]
    message: str
    data_ref: Optional[str] = None
    forecast_id: Optional[str] = None
    summary: dict[str, Any] = Field(default_factory=dict)
    ui_blocks: list[UIBlock] = Field(default_factory=list)

    def to_llm_str(self) -> str:
        """Compact string representation for the LLM context.

        The LLM only receives this text and summary stats, never raw data rows.
        """
        parts = [f"Status: {self.status}", f"Message: {self.message}"]
        if self.data_ref:
            parts.append(f"data_ref: {self.data_ref}")
        if self.forecast_id:
            parts.append(f"forecast_id: {self.forecast_id}")
        if self.summary:
            parts.append("Summary Statistics:")
            for k, v in self.summary.items():
                if isinstance(v, float):
                    parts.append(f"  - {k}: {v:.2f}")
                else:
                    parts.append(f"  - {k}: {v}")
        return "\n".join(parts)
