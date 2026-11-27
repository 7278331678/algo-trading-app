# Algo Trading App

Multi-agent AI intraday trading system for a Zerodha account: Nifty 100 universe,
6-gate signal pipeline, paper and live modes, local (Docker Desktop Kubernetes)
and cloud (AWS) deployment.

**Status: Phase 1 (data layer).** Paper mode only. No strategy code yet.

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

## Market data (Phase 1)

1. Log in to Zerodha each morning: http://localhost:8100/auth/kite/login
2. First time only, download history: double-click **`backfill.bat`** (about 10-20 minutes).
   It is resumable: if it stops, run it again and it continues where it left off.
   - only some stocks: `backfill.bat --symbols INFY TCS`
   - only some intervals: `backfill.bat --intervals day 15minute`
   - re-download after a split/bonus: `backfill.bat --refresh --symbols XYZ`
3. After that it runs itself (Kubernetes CronJobs, IST):
   - `daily-update` 16:05 Mon-Fri: today's candles, purge of old data, quality checks
   - `universe-sync` 08:15 Sunday: Nifty 100 list, Kite tokens, sectors
4. Check results: http://localhost:8100/data/status, `/data/universe`, `/data/quality`

| Interval | Kept for |
|---|---|
| 1-min | 90 days |
| 5-min, 15-min | 2 years |
| daily | 3 years |

If the NSE site blocks the automatic download of the Nifty 100 list, open
https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv in your browser, save it as
`backend/app/data/ind_nifty100list.csv`, and run `start-local.bat` again.

Backtests use point-in-time index membership. Fill `backend/app/data/universe_history.csv`
with past index changes (NSE rebalancing press releases) to remove survivorship bias; until then
the first sync assumes today's members were members for the whole data window.

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
