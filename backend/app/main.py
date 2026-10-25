"""FastAPI application entrypoint for Commodity Price Forecaster backend."""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage

from app.agent.graph import create_agent_graph
from app.api.routes import router as api_router
from app.core.cache import DataCache, global_cache
from app.core.data_access import DataAccess
from app.core.forecaster import ChronosForecaster
from app.core.risk import RiskChecker
from app.core.semantic_layer import SemanticLayer
from app.core.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

# Load environment variables from .env
load_dotenv()


class FallbackOfflineLLM:
    """Deterministic fallback LLM used when GROQ_API_KEY is not configured."""

    def bind_tools(self, tools: Any) -> "FallbackOfflineLLM":
        return self

    def invoke(self, messages: Any) -> AIMessage:
        last_msg = str(messages[-1].content).lower()
        if "forecast" in last_msg or "predict" in last_msg:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tool_forecast_price",
                        "args": {
                            "target_series_id": "onion_modal_price_maharashtra",
                            "horizon_days": 15,
                        },
                        "id": "call_auto_fc",
                    }
                ],
            )
        elif "catalog" in last_msg or "data" in last_msg:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tool_list_available_data",
                        "args": {},
                        "id": "call_auto_cat",
                    }
                ],
            )
        else:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tool_get_series",
                        "args": {
                            "series_id": "onion_modal_price_maharashtra",
                            "start_date": "2026-08-01",
                            "end_date": "2026-10-15",
                            "granularity": "daily",
                        },
                        "id": "call_auto_series",
                    }
                ],
            )


class AppState:
    """Container for application singletons and dependencies."""

    def __init__(self) -> None:
        self.settings: Settings = Settings()
        backend_root = Path(__file__).resolve().parent.parent
        self.db_path = backend_root / "data" / "duckdb" / "commodities.duckdb"
        self.semantic_path = backend_root / "app" / "config" / "semantic_layer.yaml"
        self.thresholds_path = backend_root / "app" / "config" / "thresholds.yaml"

        self.semantic_layer: SemanticLayer = SemanticLayer(self.semantic_path)
        self.data_access: DataAccess = DataAccess(self.db_path)
        self.cache: DataCache = global_cache
        self.risk_checker: RiskChecker = RiskChecker(self.thresholds_path)
        self._forecaster: Optional[ChronosForecaster] = None
        self._agent_graph: Any = None

    @property
    def forecaster(self) -> ChronosForecaster:
        if self._forecaster is None:
            self.init_services()
        return self._forecaster

    @property
    def agent_graph(self) -> Any:
        if self._agent_graph is None:
            self.init_services()
        return self._agent_graph

    def init_services(self) -> None:
        """Initialize Chronos forecaster and LangGraph agent."""
        if self._forecaster is None:
            logger.info("Initializing ChronosForecaster...")
            self._forecaster = ChronosForecaster.get_instance(
                model_name=self.settings.get("forecast.default_model", "amazon/chronos-2"),
                device="cpu",
                timeout_seconds=self.settings.get("forecast.timeout_seconds", 120),
            )

        if self._agent_graph is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key and groq_api_key.strip():
                try:
                    from langchain_groq import ChatGroq

                    llm_model = self.settings.get("llm.model", "openai/gpt-oss-20b")
                    logger.info("Connecting to Groq API with model: %s", llm_model)
                    llm = ChatGroq(
                        model=llm_model,
                        temperature=self.settings.get("llm.temperature", 0.1),
                        api_key=groq_api_key.strip(),
                    )
                except Exception as e:
                    logger.warning("Failed to initialize ChatGroq: %s. Using FallbackOfflineLLM.", e)
                    llm = FallbackOfflineLLM()
            else:
                logger.info("GROQ_API_KEY not set. Using FallbackOfflineLLM for local demonstration.")
                llm = FallbackOfflineLLM()

            self._agent_graph = create_agent_graph(
                llm=llm,
                semantic_layer=self.semantic_layer,
                data_access=self.data_access,
                cache=self.cache,
                forecaster=self._forecaster,
                risk_checker=self.risk_checker,
                as_of_date="2026-10-15",
                max_iterations=self.settings.get("agent.max_iterations", 6),
            )
            logger.info("LangGraph agent workflow compiled successfully.")


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup and shutdown lifecycle management."""
    logger.info("Starting Commodity Price Forecaster API...")
    app_state.init_services()
    yield
    logger.info("Shutting down Commodity Price Forecaster API.")


app = FastAPI(
    title="Commodity Price Forecasting Agent API",
    version="0.1.0",
    description="Production-grade API providing historical commodity analysis, Chronos-2 multi-horizon forecasting, and price risk alerts.",
    lifespan=lifespan,
)

# Enable CORS (Section 12)
cors_origins = app_state.settings.get("api.cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include versioned API router
app.include_router(api_router)


@app.get("/")
def get_root():
    return {
        "message": "Commodity Price Forecasting Agent API is active.",
        "version": "0.1.0",
        "endpoints": {
            "chat": "/api/v1/chat",
            "health": "/api/v1/health",
            "catalog": "/api/v1/catalog",
            "data": "/api/v1/data/{data_ref}",
            "forecast": "/api/v1/forecast/{forecast_id}",
            "docs": "/docs",
        },
    }
