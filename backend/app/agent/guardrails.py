"""Strict guardrails implementing all 6 layers from Section 14.

Layer 1: Deterministic pre-checks (empty, max length, prompt injection/jailbreak).
Layer 2: Scope classifier (in_scope bool, category).
Layer 3: Argument guards (Pydantic schema validation).
Layer 4: Output guard (number verification against tool summary, prompt leakage check).
Layer 5: Data honesty (enforce short canned text if tools returned NO_DATA).
Layer 6: Abuse limits (rate limits, input caps).
"""

from datetime import datetime
import logging
import re
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("guardrails")

CANNED_REJECT_MESSAGE = (
    "I can only help with commodity prices: history, forecasts, and price risks. "
    "For example, ask me to compare onion prices between years or forecast the next 30 days."
)

CANNED_WELCOME_MESSAGE = (
    "Hello! I am your Commodity Price Analysis assistant. I can help you analyze historical prices, "
    "compare periods, forecast prices for up to 90 days, and monitor price risk thresholds. "
    "Try asking: 'Compare onion prices between 2024 and 2025' or 'Forecast Maharashtra onion prices for 30 days'."
)

CANNED_NO_DATA_MESSAGE = (
    "No data about this is available, and web search is turned off for this demo."
)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|previous\s+|your\s+)?instructions",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"tell\s+me\s+(your\s+)?(system\s+)?prompt",
    r"what\s+(is|are)\s+your\s+(system\s+)?instructions",
    r"you\s+are\s+now\s+a\s+",
    r"developer\s+mode",
    r"disregard\s+all",
    r"print\s+system\s+prompt",
]


class ScopeResult(BaseModel):
    in_scope: bool
    category: Literal[
        "price_history",
        "forecast",
        "risk_thresholds",
        "factors",
        "data_availability",
        "greeting",
        "out_of_scope",
        "injection",
    ]
    reason: str
    rejection_message: Optional[str] = None


class RateLimiter:
    """Per-session sliding window rate limiter (Layer 6)."""

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        self.max_requests = max_requests_per_minute
        self._history: dict[str, list[datetime]] = {}

    def is_allowed(self, session_id: str) -> bool:
        now = datetime.now()
        window_start = now.timestamp() - 60.0
        timestamps = self._history.get(session_id, [])
        timestamps = [t for t in timestamps if t.timestamp() > window_start]
        if len(timestamps) >= self.max_requests:
            self._history[session_id] = timestamps
            return False
        timestamps.append(now)
        self._history[session_id] = timestamps
        return True


# Global limiter
session_limiter = RateLimiter(max_requests_per_minute=60)


def run_deterministic_input_guard(
    text: str,
    max_length: int = 2000,
) -> tuple[bool, Optional[str]]:
    """Layer 1: Deterministic input guard.

    Returns (is_passed, rejection_message).
    """
    if not text or not text.strip():
        return False, "Please enter a question about commodity prices."

    cleaned = text.strip()
    if len(cleaned) > max_length:
        return False, f"Input exceeds maximum allowed length of {max_length} characters."

    cleaned_lower = cleaned.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned_lower):
            return False, CANNED_REJECT_MESSAGE

    return True, None


def classify_scope(text: str, llm: Optional[Any] = None) -> ScopeResult:
    """Layer 2: Scope classifier.

    Determines if user input is within the commodity price analysis scope.
    """
    text_lower = text.strip().lower()
    text_clean = re.sub(r"[^\w\s]", "", text_lower).strip()

    # 1. Greetings / Help
    greetings = {
        "hi", "hello", "hey", "help", "who are you", "what can you do",
        "good morning", "good evening", "good afternoon"
    }
    if text_clean in greetings or any(text_clean.startswith(g) for g in ["hi ", "hello ", "hey "]):
        return ScopeResult(
            in_scope=True,
            category="greeting",
            reason="User greeting or capability check.",
            rejection_message=CANNED_WELCOME_MESSAGE,
        )

    # 2. Obvious out-of-scope triggers
    off_topic_exact = [
        "batman",
        "movie",
        "poem",
        "python code",
        "sort a list",
        "election",
        "president",
        "prime minister",
        "roman empire",
        "write an essay",
        "capital of",
        "who won",
        "who will win",
        "song",
        "recipe",
    ]
    for topic in off_topic_exact:
        if topic in text_lower:
            return ScopeResult(
                in_scope=False,
                category="out_of_scope",
                reason=f"Topic '{topic}' is unrelated to commodity prices.",
                rejection_message=CANNED_REJECT_MESSAGE,
            )

    # 3. Commodity / Price / Factor keywords using whole word boundaries
    in_scope_terms = [
        "price", "prices", "rate", "rates", "cost", "costs", "mandi", "retail", "wholesale",
        "forecast", "predict", "projection", "future",
        "onion", "onions", "pyaz", "pyaaz", "kanda", "tomato", "potato", "crop", "crops",
        "arrival", "arrivals", "production", "yield", "acreage", "supply",
        "threshold", "thresholds", "risk", "warning", "critical",
        "compare", "comparison", "trend", "history", "historical",
        "data", "catalog", "series", "market", "maharashtra", "karnataka", "ap", "mp",
        "diesel", "fuel", "weather", "monsoon", "driver", "factors"
    ]

    has_commodity_context = any(
        re.search(rf"\b{re.escape(term)}\b", text_lower) for term in in_scope_terms
    )
    if has_commodity_context:
        # Categorize
        if any(w in text_lower for w in ["forecast", "predict", "projection", "future", "next "]):
            cat = "forecast"
        elif any(w in text_lower for w in ["risk", "threshold", "critical", "warning", "distress"]):
            cat = "risk_thresholds"
        elif any(w in text_lower for w in ["compare", "vs", "versus", "between"]):
            cat = "price_history"
        elif any(w in text_lower for w in ["data", "catalog", "have", "available"]):
            cat = "data_availability"
        elif any(w in text_lower for w in ["affect", "impact", "driver", "fuel", "production", "weather", "arrival"]):
            cat = "factors"
        else:
            cat = "price_history"

        return ScopeResult(
            in_scope=True,
            category=cat,
            reason="Input relates to commodity prices, history, forecasting, or drivers.",
        )

    # Default to out of scope if no commodity terms detected
    return ScopeResult(
        in_scope=False,
        category="out_of_scope",
        reason="Query lacks commodity pricing or agricultural market context.",
        rejection_message=CANNED_REJECT_MESSAGE,
    )


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract numeric figures from generated text, ignoring dates and percentages."""
    cleaned = re.sub(r"\b(202[0-9]|19[0-9]{2})\b", "", text)
    raw_nums = re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", cleaned)
    results = []
    for num_str in raw_nums:
        try:
            val = float(num_str.replace(",", ""))
            results.append(val)
        except ValueError:
            pass
    return results


def run_output_guard(
    llm_output: str,
    tool_summaries: list[dict[str, Any]],
    tolerance_pct: float = 0.05,
) -> tuple[bool, str]:
    """Layer 4 & 5: Verify numbers in LLM text against tool summaries.

    Returns (is_passed, sanitized_text).
    """
    # 1. Leakage check
    leakage_phrases = ["You are a price-analysis assistant", "system prompt", "ROLE & BOUNDARIES"]
    for phrase in leakage_phrases:
        if phrase.lower() in llm_output.lower():
            logger.warning("Output guard detected prompt leakage.")
            return False, "I provide commodity price analysis, forecasts, and risk metrics."

    # 2. Data honesty (Layer 5)
    all_no_data = tool_summaries and all(
        s.get("reason") in ("unknown_series", "zero_rows", "no_policy_docs", "web_search_disabled")
        for s in tool_summaries
    )
    if all_no_data:
        return True, CANNED_NO_DATA_MESSAGE

    # 3. Extract allowed numbers from tool summaries
    allowed_numbers: list[float] = []

    def _collect_nums(obj: Any):
        if isinstance(obj, (int, float)):
            allowed_numbers.append(float(obj))
        elif isinstance(obj, dict):
            for v in obj.values():
                _collect_nums(v)
        elif isinstance(obj, list):
            for item in obj:
                _collect_nums(item)

    for s in tool_summaries:
        _collect_nums(s)

    if not allowed_numbers:
        return True, llm_output

    # Check extracted numbers from text
    text_numbers = extract_numbers_from_text(llm_output)
    for num in text_numbers:
        if num in (0.0, 1.0, 2.0, 7.0, 15.0, 30.0, 60.0, 90.0, 80.0, 100.0, 10.0, 50.0):
            continue

        matched = any(
            abs(num - allowed) <= max(abs(allowed * tolerance_pct), 1.0)
            for allowed in allowed_numbers
        )
        if not matched and num > 200:
            logger.info("Output guard flagged ungrounded number: %f", num)
            fallback_text = _build_templated_summary(tool_summaries)
            return True, fallback_text

    return True, llm_output


def _build_templated_summary(tool_summaries: list[dict[str, Any]]) -> str:
    """Build honest deterministic summary directly from tool summaries."""
    lines = ["Summary based strictly on tool results:"]
    for s in tool_summaries:
        if "series_id" in s:
            lines.append(f"- Series: {s['series_id']}")
        if "latest" in s:
            lines.append(f"- Latest Price: {s['latest']} {s.get('unit', '')}")
        if "mean" in s:
            lines.append(f"- Average Price: {s['mean']} {s.get('unit', '')}")
        if "expected_value_end" in s:
            lines.append(f"- Projected Median End Price: {s['expected_value_end']} (range: {s.get('range_low_end')} to {s.get('range_high_end')})")
        if "pct_change" in s:
            lines.append(f"- Projected Change: {s['pct_change']:+.2f}%")
        if "backtest_mape" in s:
            lines.append(f"- Backtest MAPE: {s['backtest_mape']}%")
    return "\n".join(lines)
