# APIs in Base

All endpoints are under `/api/v1`.

| Method | Endpoint | What it does |
|---|---|---|
| GET | `/health` | Checks DB connection and model load status |
| GET | `/catalog` | Returns available commodity series list |
| POST | `/chat` | Main agent endpoint; takes user message, runs LangGraph, returns UI blocks |
| GET | `/data/{data_ref}` | Fetches historical data points by reference handle |
| GET | `/forecast/{forecast_id}` | Fetches forecast results, backtest, and risk alerts by forecast ID |
| POST | `/session/reset` | Clears session memory/state |
