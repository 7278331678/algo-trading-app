"""End-to-end feature build on synthetic candles with a fake database and object store."""

import json
import pickle
import sys
import types
from datetime import date, datetime

import pandas as pd
import pytest

from app.core.market_clock import IST, MarketClock
from tests.test_features import candles, resample

TOK = {"NIFTY 50": 1, "INDIA VIX": 2, "NIFTY IT": 3, "INFY": 10, "TCS": 11}


class FakeConn:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        q = str(sql)
        if "segment='INDICES'" in q and "tradingsymbol FROM instruments" in q:
            return FakeResult([(TOK[n], n) for n in ("NIFTY 50", "INDIA VIX", "NIFTY IT")])
        if "FROM universe_members" in q:
            return FakeResult([("INFY", "NIFTY IT", date(2023, 1, 1), None),
                               ("TCS", "NIFTY IT", date(2023, 1, 1), None)])
        if "price_jump" in q:
            return FakeResult([])
        raise AssertionError(q)


class FakeResult(list):
    def all(self):
        return list(self)


class FakeEngine:
    def connect(self):
        return FakeConn()


@pytest.fixture
def env(monkeypatch):
    sys.modules.setdefault("sqlalchemy", types.SimpleNamespace(text=lambda s: s))
    from app.features import pipeline
    from app.marketdata import universe

    m5 = {t: candles(days=60, start="2026-08-17") for t in TOK.values()}
    data = {}
    for t, f in m5.items():
        daily = f.groupby("day").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                     close=("close", "last"), volume=("volume", "sum")).reset_index()
        daily["ts"] = pd.to_datetime(daily["day"]).dt.tz_localize("Asia/Kolkata")
        data[(t, "5minute")] = f
        data[(t, "15minute")] = resample(f, "15min")
        data[(t, "day")] = pipeline.build.frame(daily.drop(columns=["day"]))
    monkeypatch.setattr(pipeline, "_load", lambda engine, t, i, since: data[(t, i)].copy())
    monkeypatch.setattr(universe, "tokens_to_fetch", lambda e, idx, since: {v: k for k, v in TOK.items()})
    written = {}

    def to_parquet(self, buf, index=False):
        buf.write(pickle.dumps(self))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)
    settings = types.SimpleNamespace(retention_5minute_days=730, retention_day_days=1095, universe_index="NIFTY 100")
    return pipeline, settings, written


def test_full_build(env):
    pipeline, settings, written = env
    stats = pipeline.build_universe_features(settings, FakeEngine(), MarketClock(),
                                             lambda k, b: written.__setitem__(k, b),
                                             now=datetime(2026, 11, 27, 18, 0, tzinfo=IST))
    assert stats["stocks"] == 2 and stats["labelled"] > 0
    assert set(written) == {"features/v1/5minute/INFY.parquet", "features/v1/5minute/TCS.parquet",
                            "features/v1/manifest.json"}
    df = pickle.loads(written["features/v1/5minute/INFY.parquet"])
    from app.features import build, labels
    for col in build.ALL_FEATURES + labels.LABEL_COLUMNS + ["exclude", "in_universe"]:
        assert col in df.columns, col
    usable = df[~df["exclude"] & df["y_long"].notna()]
    assert len(usable) > 1000
    # context columns are actually filled for usable rows
    for col in ["nifty_ret_6", "vix", "sector_ret_6", "breadth_above_vwap", "ema50_15m_dist", "gap_pct"]:
        assert usable[col].notna().mean() > 0.95, col
    assert df["exclude_reason"].eq("warmup").any()
    manifest = json.loads(written["features/v1/manifest.json"])
    assert manifest["features"] == build.ALL_FEATURES and len(manifest["code_hash"]) == 16
    # labels end before our exit time: in the CAS period no entry after 14:30 decision
    dec = (df.loc[df["y_long"].notna(), "decision_time"]).dt.time
    assert dec.max() <= pd.Timestamp("14:30").time()
