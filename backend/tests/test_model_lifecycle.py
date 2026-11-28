"""Finishing Phase 4: planted-signal check, MLflow tracking, registry decisions, forward evaluation."""

from datetime import time

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from app.backtest.engine import BacktestConfig
from app.core.market_clock import MarketClock
from app.features import build
from app.models import dataset, registry, signal_check, study, tracking, walkforward

from .test_models import synthetic_market


# ------------------------------------------------------------------ planted signal
def test_expected_auc_values():
    assert signal_check.expected_auc(0) == 0.5
    assert signal_check.expected_auc(0.09) == pytest.approx(0.551, abs=0.001)
    assert signal_check.expected_auc(0.2) == pytest.approx(0.611, abs=0.001)


@pytest.mark.parametrize("s", [0.0, 0.09, 0.2])
def test_planted_feature_carries_exactly_the_promised_information(s):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"t_long": rng.integers(0, 2, 200_000).astype(float),
                       "t_short": rng.integers(0, 2, 200_000).astype(float)})
    df.loc[::50, "t_long"] = np.nan
    signal_check.plant(df, s)
    ok = df["t_long"].notna()
    assert df.loc[~ok, "planted_long"].isna().all()
    auc = roc_auc_score(df.loc[ok, "t_long"], df.loc[ok, "planted_long"])
    assert auc == pytest.approx(signal_check.expected_auc(s), abs=0.006)


def test_judge_rules():
    assert signal_check.judge(0.0, 0.505, "no_edge", None)["pass"]
    assert not signal_check.judge(0.0, 0.56, "no_edge", None)["pass"]            # found signal in noise
    assert not signal_check.judge(0.0, 0.50, "edge_candidate", None)["pass"]     # called noise an edge
    assert signal_check.judge(0.2, 0.60, "edge_candidate", {"net_pnl": 500})["pass"]
    assert not signal_check.judge(0.2, 0.60, "no_edge", {"net_pnl": -5})["pass"]  # strong edge must pay
    assert not signal_check.judge(0.09, 0.51, "no_edge", None)["pass"]
    assert signal_check.judge(0.09, 0.54, "no_edge", None)["pass"]


def test_signal_check_end_to_end_on_a_market_with_no_edge():
    df = synthetic_market(n_sym=6, days=330, seed=21, drift=0.0)   # pure random walk: nothing to find
    df["or_breakout"] = np.random.default_rng(3).normal(0, 1, len(df)).astype("float32")
    res = signal_check.run(df, MarketClock(), BacktestConfig(capital=100_000), "daily_0.6x1.2",
                           dataset.CostModel(slippage_bps_per_side=2.0), ("logit",), [time(9, 45)],
                           strengths=(0.0, 0.5))
    null, strong = res["results"]
    assert abs(null["measured_auc"] - 0.5) < 0.04 and null["verdict"] != "edge_candidate"
    assert strong["measured_auc"] > 0.66 and strong["holdout"]["net_pnl"] > 0
    assert res["weakest_profitable_strength"] == 0.5


# ------------------------------------------------------------------ study: final model matches decision mode
def test_final_mask_uses_only_decision_rows():
    df = synthetic_market(n_sym=3, days=300, seed=4)
    res = study.run_study(df, MarketClock(), BacktestConfig(capital=100_000), "daily_0.6x1.2",
                          dataset.CostModel(), ("logit",), decision_times=[time(9, 45)])
    rows = res["_df"].loc[res["_final_mask"]]
    assert len(rows) and (rows["decision_time"].dt.strftime("%H:%M") == "09:45").all()
    assert study.trained_until(res) == max(rows["day"])


# ------------------------------------------------------------------ MLflow over REST
def _fake_mlflow(calls, fail=False):
    def send(method, path, body, ctype):
        calls.append((method, path.split("?")[0]))
        if fail:
            return 503, None
        if "experiments/get-by-name" in path:
            return 404, None
        if path.endswith("experiments/create"):
            return 200, {"experiment_id": "3"}
        if path.endswith("runs/create"):
            return 200, {"run": {"info": {"run_id": "abc", "artifact_uri": "mlflow-artifacts:/3/abc/artifacts"}}}
        return 200, {}
    return send


RESULT = {"verdict": "no_edge", "barrier": "daily_0.6x1.2", "rows": 10, "trainable_rows": 8,
          "cost_round_trip_pct": 0.142, "holdout_quarters": ["2026Q3"],
          "base_win_rate_after_costs": {"long": 0.39, "short": 0.41},
          "fold_metrics": [{"quarter": "2026Q3", "side": "long", "model": "ensemble", "auc": 0.51}],
          "chosen": {"model": "hgb", "threshold": 0.6, "net_pnl": -10.0, "would_halt_on": None},
          "chosen_on_holdout": {"model": "hgb", "net_pnl": -5.0}, "_df": "not sent"}


def test_mlflow_logging_calls_the_rest_api():
    calls = []
    rid = tracking.log_training_run("http://mlflow:5000", "20261128-1", {"barrier": "daily_0.6x1.2"}, RESULT,
                                    send=_fake_mlflow(calls))
    assert rid == "abc"
    paths = [p for _, p in calls]
    assert "/api/2.0/mlflow/runs/create" in paths and "/api/2.0/mlflow/runs/update" in paths
    assert ("PUT", "/api/2.0/mlflow-artifacts/artifacts/3/abc/artifacts/summary.json") in calls


def test_mlflow_down_never_breaks_training():
    assert tracking.log_training_run("http://mlflow:5000", "r", {}, RESULT, send=_fake_mlflow([], fail=True)) is None
    assert tracking.log_training_run(None, "r", {}, RESULT) is None


def test_flatten_keeps_numbers_only():
    assert tracking.flatten({"a": {"b": 1, "c": "x", "d": float("nan")}, "e": True}) == {"a.b": 1.0, "e": 1.0}


# ------------------------------------------------------------------ registry decisions
GOOD = {"trades": 40, "net_pnl": 3000.0, "profit_factor": 1.5, "would_halt_on": None}


def test_gate():
    assert registry.gate("edge_candidate")[0] == "shadow"
    assert registry.gate("weak_positive")[0] == "rejected"
    assert registry.gate("no_edge")[0] == "rejected"


def test_shadow_waits_then_decides_once():
    assert registry.decide_shadow(GOOD, 10, None)[0] == "wait"
    assert registry.decide_shadow(GOOD, 20, None)[0] == "promote"
    assert registry.decide_shadow({**GOOD, "net_pnl": -1.0}, 25, None)[0] == "retire"
    assert registry.decide_shadow({**GOOD, "trades": 5}, 25, None)[0] == "retire"
    assert registry.decide_shadow({**GOOD, "would_halt_on": "2026-12-01"}, 25, None)[0] == "retire"
    assert registry.decide_shadow({**GOOD, "profit_factor": 1.1}, 25, None)[0] == "retire"
    # must beat the champion over the same unseen days
    assert registry.decide_shadow(GOOD, 25, {"net_pnl": 4000.0})[0] == "retire"
    assert registry.decide_shadow(GOOD, 25, {"net_pnl": 1000.0})[0] == "promote"


def test_champion_retired_when_forward_results_go_bad():
    assert registry.decide_champion(GOOD)[0] == "keep"
    assert registry.decide_champion({**GOOD, "would_halt_on": "2026-12-01"})[0] == "retire"
    assert registry.decide_champion({**GOOD, "profit_factor": 0.6})[0] == "retire"
    assert registry.decide_champion({"trades": 3, "profit_factor": 0.2})[0] == "keep"   # too few to judge


# ------------------------------------------------------------------ forward evaluation on unseen days
def test_forward_eval_scores_only_unseen_days_and_finds_a_real_edge():
    clock = MarketClock()
    df = synthetic_market(n_sym=4, days=330, seed=8)
    prepared = study.prepare(df, clock, "daily_0.6x1.2", dataset.CostModel(slippage_bps_per_side=2.0))
    days = sorted(prepared["day"].unique())
    cutoff = days[250]
    train = prepared["day"] <= cutoff
    at = prepared["decision_time"].dt.strftime("%H:%M") == "09:45"
    mask = dataset.trainable(prepared) & train & at
    models = walkforward.fit_final(prepared, build.ALL_FEATURES, mask, ("logit",), 50_000)
    manifest = {"features": build.ALL_FEATURES, "barrier": "daily_0.6x1.2", "kinds": ["logit"],
                "spec": {"decision_times": ["09:45"], "slippage_bps": 2.0, "kinds": ["logit"]}}
    s, n, first, last = registry.forward_eval(df, clock, BacktestConfig(capital=100_000), models, manifest,
                                              {"model": "logit", "threshold": 0.55}, cutoff)
    assert first > cutoff and n == len(days) - 251
    assert s["trades"] > 20 and s["net_pnl"] > 0


def test_score_predicts_only_at_decision_times():
    df = synthetic_market(n_sym=1, days=3)

    class Const:
        def predict(self, X):
            return np.full(len(X), 0.7)

    out = registry.score(df.copy(), {"long_logit": Const(), "short_logit": Const()}, build.ALL_FEATURES,
                         ("logit",), [time(9, 45)])
    hit = out["decision_time"].dt.strftime("%H:%M") == "09:45"
    assert (out.loc[hit, "p_long"] == 0.7).all() and out.loc[~hit, "p_long"].isna().all()
