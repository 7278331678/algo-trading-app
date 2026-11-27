# Algo Trading App

Multi-agent AI intraday trading system for a Zerodha account: Nifty 100 universe,
6-gate signal pipeline, paper and live modes, local (Docker Desktop Kubernetes)
and cloud (AWS) deployment.

**Status: Phase 0 (foundation).** Paper mode only. No strategy code yet.

## Quick start (Windows, local mode)

1. Install and start: Docker Desktop (Settings > Kubernetes > Enable), Helm
   (`winget install Helm.Helm`), Ollama.
2. Copy `.env.example` to `.env` and fill it in. Use letters and digits only for passwords.
3. In your Kite Connect app (developers.kite.trade), set the redirect URL to
   `http://127.0.0.1:8100/auth/kite/callback`.
4. Double-click **`start-local.bat`**.
5. Open http://localhost:8100/auth/kite/login each morning to log in to Zerodha.

| Service | URL |
|---|---|
| API docs | http://localhost:8100/docs |
| System status | http://localhost:8100/health/system |
| MLflow | http://localhost:5000 |
| MinIO console | http://localhost:9001 |
| Mail inbox (MailHog) | http://localhost:8025 |

`stop-local.bat` stops everything and keeps data. `reset-local.bat` deletes everything, including data.

## Layout

```
backend/        FastAPI API, config, broker layer, market clock
  app/broker/   Broker interface, Kite adapter, simulator (paper/backtest/replay)
  app/core/     NSE market clock (IST, holidays, entry window)
  app/trading/  Zerodha charges
services/       hot path (Phase 6)
agents/         cool path, LangGraph (Phases 5-6)
mcp-servers/    MCP tools (Phase 5)
ml/             data, features, backtester, models (Phases 1-4)
frontend/       React dashboard (Phase 7)
mobile/         React Native app (Phase 10)
deploy/helm/    Helm chart (local now, AWS in Phase 9)
deploy/terraform/  AWS infrastructure (Phase 9)
```

## Safety rules built in from day one

- `TRADING_MODE=paper` by default. Live mode also needs `LIVE_TRADING_CONFIRM=I-UNDERSTAND-THE-RISK`.
- Every order carries a unique 20-character tag; placing the same tag twice returns the first order.
- The market clock refuses to trade in a year with no NSE holiday calendar
  (`backend/app/data/nse_holidays.json`). Add next year's list from the NSE circular each December.
- The Kite access token is stored in Redis only, expires at 06:00 IST, and is never returned by the API.
- Secrets live in `.env` locally (loaded into a Kubernetes secret) and in AWS Secrets Manager in cloud mode.

## Tests

```
cd backend
pip install -r requirements-dev.txt
pytest -q
```
