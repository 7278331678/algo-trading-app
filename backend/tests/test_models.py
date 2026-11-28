"""Phase 4: the training study finds a real (synthetic) edge, and never trains on the future."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import BacktestConfig
from app.core.market_clock import MarketClock
from app.features import build
from app.models import dataset, regime, study, walkforward


def synthetic_market(n_sym=6, start="2025-01-01", days=330, seed=5, drift=0.0012):
    """Days with a hidden trend: after 09:30 the stock drifts up or down, and `or_breakout`
    (known from 09:30) reveals the direction 70% of the time. Other features are noise."""
    rng = np.random.default_rng(seed)
    bdays = pd.bdate_range(start, periods=days)
    frames = []
    for s in range(n_sym):
        price = 500.0
        rows = []
        for d in bdays:
            trend = rng.choice([-1, 1])
            hint = trend if rng.random() < 0.7 else -trend
            t0 = d + pd.Timedelta(hours=9, minutes=15)
            n = 72  # auction stock: last candle 15:10
            drift_ = np.where(np.arange(n) >= 3, trend * drift, 0.0)
            r = drift_ + rng.normal(0, 0.0012, n)
            c = price * np.exp(np.cumsum(r))
            o = np.r_[price, c[:-1]]
            ts = pd.date_range(t0, periods=n, freq="5min", tz="Asia/Kolkata")
            df = pd.DataFrame({"ts": ts, "open": o, "close": c,
                               "high": np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0004, n))),
                               "low": np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0004, n)))})
            df["day"] = d.date()
            price = c[-1]
            rows.append(df.assign(or_breakout=np.where(np.arange(n) >= 3, float(hint), np.nan)))
        f = pd.concat(rows, ignore_index=True)
        f["symbol"] = f"S{s}"
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["decision_time"] = df["ts"] + pd.Timedelta(minutes=5)
    for c in build.ALL_FEATURES:
        if c not in df:
            df[c] = rng.normal(0, 1, len(df)).astype("float32")
    df["minutes_since_open"] = (df["ts"].dt.hour * 60 + df["ts"].dt.minute + 5 - 555).astype("float32")
    df["nifty_ret_open"], df["vix_chg"], df["breadth_above_vwap"] = 0.0, 0.0, 0.6
    df["rel_strength"] = 0.01
    df["atr14"] = (df["close"] * 0.0015).astype("float32")
    df["daily_atr_pct"] = np.float32(0.015)
    df["exclude"], df["in_universe"], df["sector_index"] = False, True, None
    return df


@pytest.fixture(scope="module")
def result():
    df = synthetic_market()
    return study.run_study(df, MarketClock(), BacktestConfig(capital=100_000), barrier="daily_0.4x0.6",
                           cost=dataset.CostModel(slippage_bps_per_side=2.0), max_train_rows=200_000)


def test_models_learn_the_hidden_signal(result):
    ens = [m for m in result["fold_metrics"] if m["model"] == "ensemble" and m["side"] == "long"]
    assert ens and all(m["auc"] > 0.6 for m in ens)
    assert all(m["top_decile_lift"] > 1.2 for m in ens)


def test_threshold_chosen_on_selection_and_holdout_profits(result):
    assert result["chosen"] is not None
    assert set(result["selection_quarters"]).isdisjoint(result["holdout_quarters"])
    h = result["chosen_on_holdout"]
    assert h["trades"] >= 30 and h["net_pnl"] > 0
    assert result["verdict"] in {"edge_candidate", "weak_positive"}


def test_cost_aware_targets(result):
    df = result["_df"]
    m = result["_mask"]
    c = dataset.CostModel(slippage_bps_per_side=2.0).round_trip
    wins = df.loc[m & (df["t_long"] == 1), "ret_long"]
    losses = df.loc[m & (df["t_long"] == 0), "ret_long"]
    assert (wins > c).all() and (losses <= c).all()


def test_walk_forward_never_trains_on_its_own_quarter_or_later(monkeypatch):
    df = synthetic_market(n_sym=2, days=200)
    df = study.runner.add_exit_times(df, MarketClock())
    dataset.relabel(df, "daily_0.4x0.6")
    dataset.add_targets(df, dataset.CostModel())
    seen = []
    real = walkforward.fit_calibrated

    def spy(kind, X, y, ts, max_rows, seed=7, calib_frac=0.15):
        seen.append(pd.Timestamp(ts.max()))
        return real(kind, X, y, ts, max_rows, seed, calib_frac)

    monkeypatch.setattr(walkforward, "fit_calibrated", spy)
    preds, _ = walkforward.run_walkforward(df, build.ALL_FEATURES, dataset.trainable(df), ("logit",), 50_000)
    test_days = pd.to_datetime(df.loc[preds.index, "day"])
    first_test_day = test_days.min()
    assert seen and min(seen).tz_convert(None) < first_test_day
    # every fold's training data ends before that fold's quarter begins
    quarters = sorted(set(pd.PeriodIndex(test_days, freq="Q")))
    per_fold = len(seen) // len(quarters)
    for i, q in enumerate(quarters):
        for t in seen[i * per_fold:(i + 1) * per_fold]:
            assert t.tz_convert(None) < q.start_time


def test_bad_day_gate():
    df = pd.DataFrame({"ts": pd.date_range("2026-11-25 09:15", periods=6, freq="5min", tz="Asia/Kolkata"),
                       "day": date(2026, 11, 25), "symbol": "A",
                       "minutes_since_open": [5, 10, 15, 20, 25, 30],
                       "nifty_ret_open": [0.0, -0.002, -0.011, -0.004, 0.0, 0.0],
                       "vix_chg": 0.0, "breadth_above_vwap": 0.6})
    flag = regime.bad_day_flag(df)
    assert flag.tolist() == [False, False, True, True, True, True]   # once bad, bad for the day


def test_save_writes_models_predictions_and_manifest(result, monkeypatch):
    import io
    import json
    import pickle
    import sys
    import types

    import joblib

    sys.modules.setdefault("sqlalchemy", types.SimpleNamespace(text=lambda q: q))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, buf, index=False: buf.write(pickle.dumps(self)))

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, q, params):
            self.params = params
            return types.SimpleNamespace(scalar_one=lambda: 42)

    class Engine:
        def begin(self):
            return Conn()

    final = walkforward.fit_final(result["_df"], build.ALL_FEATURES, result["_mask"], ("logit",), 20_000)
    written = {}
    pk = study.save(Engine(), written.__setitem__, "test-run", result, final, ("logit",))
    assert pk == 42
    assert {"models/test-run/predictions.parquet", "models/test-run/manifest.json",
            "models/test-run/long_logit.joblib", "models/test-run/short_logit.joblib"} <= set(written)
    model = joblib.load(io.BytesIO(written["models/test-run/long_logit.joblib"]))
    X = result["_df"][build.ALL_FEATURES].to_numpy(np.float32)[:10]
    p = model.predict(X)
    assert ((p >= 0) & (p <= 1)).all()
    manifest = json.loads(written["models/test-run/manifest.json"])
    assert manifest["verdict"] == result["verdict"] and manifest["features"] == build.ALL_FEATURES


def test_load_has_unique_columns(monkeypatch):
    """Regression: daily_atr_pct is both a meta column and a feature; it must be loaded once."""
    import pickle

    frame = synthetic_market(n_sym=1, days=5)
    frame["day"] = pd.to_datetime(frame["day"])
    store = {"features/v1/5minute/S0.parquet": pickle.dumps(frame)}

    def fake_read(buf, columns=None):
        df = pickle.loads(buf.getvalue())
        return df[columns]          # pandas, like parquet, returns duplicates if asked twice

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    df = dataset.load(lambda p: list(store), store.__getitem__)
    assert df.columns.is_unique
    df = study.runner.add_exit_times(df, MarketClock())
    dataset.relabel(df, "daily_0.4x0.6")      # the step that failed on real data
    assert df["ret_long"].notna().any()


def test_one_decision_per_day_mode():
    from datetime import time

    df = synthetic_market(n_sym=4, days=330, seed=9)
    res = study.run_study(df, MarketClock(), BacktestConfig(capital=100_000), barrier="daily_0.6x1.2",
                          cost=dataset.CostModel(slippage_bps_per_side=2.0), kinds=("logit",),
                          decision_times=[time(9, 45)])
    assert res["decision_times"] == ["09:45:00"]
    preds = res["_frames"]["predictions"]
    assert (preds["ts"].dt.strftime("%H:%M") == "09:40").all()     # candle 09:40-09:45 only
    h = res["chosen_on_holdout"]
    assert h is not None and h["trades"] > 0 and h["trades_per_day"] <= 4
