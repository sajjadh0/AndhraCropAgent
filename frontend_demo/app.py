"""Streamlit demo frontend for Commodity Price Forecaster.

Strict adherence to Section 4 & Hard Rule #4:
- The frontend talks ONLY to the HTTP API (http://localhost:8000/api/v1).
- The frontend contains NO backend database, model, or calculation logic.
- Renders ordered UI blocks: text, chart, risk_alert, forecast_summary, notice.
"""

import os
from typing import Any, Optional
import httpx
import plotly.graph_objects as go
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="AgriCast: Commodity Price Forecaster",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY", "")


def get_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY.strip():
        headers["X-API-Key"] = API_KEY.strip()
    return headers


def check_api_health() -> Optional[dict[str, Any]]:
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", headers=get_headers(), timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_catalog() -> dict[str, Any]:
    try:
        resp = httpx.get(f"{API_BASE_URL}/catalog", headers=get_headers(), timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        st.sidebar.error(f"Failed to fetch catalog: {e}")
    return {"commodities": ["onion"], "series": []}


def send_chat_message(session_id: str, message: str) -> dict[str, Any]:
    payload = {"session_id": session_id, "message": message}
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            headers=get_headers(),
            timeout=120.0,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return {
                "blocks": [
                    {
                        "type": "text",
                        "content": f"⚠️ Server Error ({resp.status_code}): {resp.text}",
                    }
                ]
            }
    except Exception as e:
        return {
            "blocks": [
                {
                    "type": "text",
                    "content": f"⚠️ Unable to reach backend API at `{API_BASE_URL}`. Ensure FastAPI server is running (`uvicorn app.main:app --port 8000`). Error details: {e}",
                }
            ]
        }


def reset_server_session(session_id: str) -> bool:
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/session/reset",
            json={"session_id": session_id},
            headers=get_headers(),
            timeout=5.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ==========================================
# UI Block Renderers
# ==========================================

def render_chart_block(block: dict[str, Any]) -> None:
    chart_type = block.get("chart_type", "line")
    title = block.get("title", "Price Chart")
    x_label = block.get("x_label", "Date")
    y_label = block.get("y_label", "Price (Rs/Qtl)")
    series_list = block.get("series", [])

    fig = go.Figure()

    if chart_type == "forecast_band":
        # Multi-quantile forecast fan chart
        # Look for historical, p50, and band series
        hist_series = next((s for s in series_list if s.get("name") in ("Historical", "History")), None)
        p50_series = next((s for s in series_list if "P50" in s.get("name", "") or "Forecast" in s.get("name", "")), None)
        p10_series = next((s for s in series_list if "P10" in s.get("name", "") or "Lower" in s.get("name", "")), None)
        p90_series = next((s for s in series_list if "P90" in s.get("name", "") or "Upper" in s.get("name", "")), None)

        # Plot historical
        if hist_series:
            fig.add_trace(
                go.Scatter(
                    x=hist_series.get("x", []),
                    y=hist_series.get("y", []),
                    mode="lines",
                    name="Historical Modal Price",
                    line=dict(color="#1f77b4", width=2),
                )
            )

        # Plot p10 lower bound
        if p10_series:
            fig.add_trace(
                go.Scatter(
                    x=p10_series.get("x", []),
                    y=p10_series.get("y", []),
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Plot p90 upper bound with fill
        if p90_series:
            fig.add_trace(
                go.Scatter(
                    x=p90_series.get("x", []),
                    y=p90_series.get("y", []),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(255, 127, 14, 0.22)",
                    name="80% Uncertainty Band (P10 - P90)",
                )
            )

        # Plot p50 expected forecast
        if p50_series:
            fig.add_trace(
                go.Scatter(
                    x=p50_series.get("x", []),
                    y=p50_series.get("y", []),
                    mode="lines+markers",
                    name="Chronos Forecast (P50 Expected)",
                    line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                    marker=dict(size=4),
                )
            )

        # Fallback if specific names not identified
        if not hist_series and not p50_series:
            for s in series_list:
                fig.add_trace(
                    go.Scatter(
                        x=s.get("x", []),
                        y=s.get("y", []),
                        mode="lines",
                        name=s.get("name", "Series"),
                    )
                )

    elif chart_type in ("line", "line_compare"):
        for s in series_list:
            fig.add_trace(
                go.Scatter(
                    x=s.get("x", []),
                    y=s.get("y", []),
                    mode="lines",
                    name=s.get("name", "Series"),
                    line=dict(width=2),
                )
            )

    elif chart_type == "bar":
        for s in series_list:
            fig.add_trace(
                go.Bar(
                    x=s.get("x", []),
                    y=s.get("y", []),
                    name=s.get("name", "Series"),
                )
            )

    # Reference threshold lines (if provided)
    ref_lines = block.get("reference_lines", [])
    for ref in ref_lines:
        fig.add_hline(
            y=ref["value"],
            line_dash="dot",
            line_color="red" if "danger" in ref.get("label", "").lower() or "high" in ref.get("label", "").lower() else "green",
            annotation_text=ref.get("label", ""),
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"chart_{id(block)}")


def render_risk_alert_block(block: dict[str, Any]) -> None:
    severity = str(block.get("severity", "MEDIUM")).upper()
    threshold_name = block.get("threshold_name", "Risk Threshold")
    threshold_val = block.get("threshold_value", 0.0)
    pred_val = block.get("predicted_value", 0.0)
    crossing_date = block.get("crossing_date", "Horizon window")
    action_note = block.get("actionable_advice") or block.get("message", "")

    color_map = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
    }
    badge = color_map.get(severity, "⚠️")

    st.markdown(
        f"""
        <div style="border-left: 5px solid {'#d32f2f' if severity in ('CRITICAL', 'HIGH') else '#fbc02d'}; 
                    padding: 12px 16px; background-color: {'#ffebee' if severity in ('CRITICAL', 'HIGH') else '#fffde7'}; 
                    border-radius: 4px; margin: 10px 0;">
            <div style="font-weight: bold; font-size: 1.05rem; color: #212121;">
                {badge} Price Risk Alert: {severity} - {threshold_name}
            </div>
            <div style="font-size: 0.92rem; color: #424242; margin-top: 4px;">
                <strong>Threshold:</strong> ₹{threshold_val:,.1f}/Qtl &nbsp;|&nbsp; 
                <strong>Predicted:</strong> ₹{pred_val:,.1f}/Qtl &nbsp;|&nbsp; 
                <strong>Crossing Date:</strong> {crossing_date}
            </div>
            <div style="font-size: 0.88rem; color: #616161; margin-top: 6px;">
                <em>Advisory:</em> {action_note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_summary_block(block: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Current Price",
            value=f"₹{block.get('start_price', 0.0):,.1f}",
            help="Latest observed price in target series",
        )
    with c2:
        pct = block.get("pct_change", 0.0)
        delta_str = f"{pct:+.1f}%"
        st.metric(
            label=f"Forecast P50 ({block.get('horizon_days', 15)}d)",
            value=f"₹{block.get('end_price_p50', 0.0):,.1f}",
            delta=delta_str,
            delta_color="inverse",  # Higher onion price is inflation risk
        )
    with c3:
        p10 = block.get("end_price_p10", 0.0)
        p90 = block.get("end_price_p90", 0.0)
        st.metric(
            label="80% Uncertainty Band",
            value=f"₹{p10:,.0f} - ₹{p90:,.0f}",
            help="10th percentile (lower) to 90th percentile (upper)",
        )
    with c4:
        mape = block.get("backtest_mape")
        mae = block.get("backtest_mae")
        mape_str = f"{mape:.1f}%" if mape is not None else "N/A"
        mae_str = f"₹{mae:.1f}" if mae is not None else "N/A"
        st.metric(
            label="Holdout Backtest",
            value=mape_str,
            help=f"Mean Absolute Percentage Error over holdout window. MAE: {mae_str}",
        )


def render_notice_block(block: dict[str, Any]) -> None:
    msg = block.get("message", "")
    level = block.get("level", "info").lower()
    if level == "warning":
        st.warning(f"ℹ️ {msg}")
    elif level == "error":
        st.error(f"❌ {msg}")
    else:
        st.info(f"ℹ️ {msg}")


def render_blocks(blocks: list[dict[str, Any]]) -> None:
    for block in blocks:
        b_type = block.get("type")
        if b_type == "text":
            st.markdown(block.get("content", ""))
        elif b_type == "forecast_summary":
            render_forecast_summary_block(block)
        elif b_type == "risk_alert":
            render_risk_alert_block(block)
        elif b_type == "chart":
            render_chart_block(block)
        elif b_type == "notice":
            render_notice_block(block)


# ==========================================
# Main Streamlit App Layout
# ==========================================

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"

# Sidebar Configuration
with st.sidebar:
    st.title("🌾 AgriCast Controls")
    
    # API Health Status
    health = check_api_health()
    if health and health.get("status") == "healthy":
        st.success("🟢 Backend API: Healthy")
    else:
        st.error("🔴 Backend API: Disconnected")
        st.caption(f"Make sure backend runs on `{API_BASE_URL}`")

    st.markdown("---")
    
    # Series Catalog
    catalog = fetch_catalog()
    series_options = {s["id"]: f"{s['description']} ({s['state']})" for s in catalog.get("series", [])}
    
    st.subheader("Data Catalog")
    selected_series_id = st.selectbox(
        "Available Series",
        options=list(series_options.keys()) if series_options else ["onion_modal_price_maharashtra"],
        format_func=lambda sid: series_options.get(sid, sid),
    )
    
    # Pre-filled Quick Questions
    st.subheader("Quick Prompts")
    quick_prompts = [
        "Forecast onion prices in Maharashtra for 15 days",
        "Forecast onion prices in Maharashtra for 30 days using arrivals",
        "Compare onion modal prices between 2025 and 2026 in Maharashtra",
        "What data do you have in the catalog?",
        "Show recent modal prices in Maharashtra",
        "When was Batman released?",
    ]
    
    prompt_to_submit = None
    for q in quick_prompts:
        if st.button(q, key=f"btn_{q}"):
            prompt_to_submit = q

    st.markdown("---")
    
    # Reset Session
    if st.button("🔄 Reset Conversation State", use_container_width=True):
        reset_server_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

    st.caption("AgriCast v0.1.0 • Amazon Chronos-2 • LangGraph")


# Main Page Header
st.title("🧅 Commodity Price Forecasting Agent")
st.caption("AI-powered commodity market intelligence, historical time-series analytics, and Chronos-2 probabilistic forecasting.")

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_blocks(msg.get("blocks", []))

# Handle User Input (either from chat input or quick prompt button)
user_prompt = st.chat_input("Ask a question about onion prices, historical trends, or future forecasts...")
if prompt_to_submit:
    user_prompt = prompt_to_submit

if user_prompt:
    # 1. Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Call backend API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing market data and generating forecast..."):
            response = send_chat_message(st.session_state.session_id, user_prompt)
            blocks = response.get("blocks", [])
            render_blocks(blocks)

    # 3. Save assistant response blocks in state
    st.session_state.messages.append({"role": "assistant", "blocks": blocks})
