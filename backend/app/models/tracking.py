"""Experiment tracking in MLflow over its REST API (stdlib only: no mlflow package in the image).

Tracking is best-effort: if MLflow is down, training still finishes and is still recorded in
Postgres and object storage; we just log a warning.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

log = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def flatten(d: dict, prefix: str = "") -> dict[str, float]:
    """Nested dict -> {"a.b": number} for numeric leaves only."""
    out: dict[str, float] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, int | float) and v == v:   # skip NaN
            out[key] = float(v)
    return out


# send(method, path, body_bytes, content_type) -> (status, parsed_json_or_None)
Sender = Callable[[str, str, bytes | None, str], tuple[int, dict | None]]


def urllib_sender(base: str, timeout: float = 15.0) -> Sender:
    def send(method: str, path: str, body: bytes | None, ctype: str) -> tuple[int, dict | None]:
        req = urllib.request.Request(base.rstrip("/") + path, data=body, method=method,
                                     headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - fixed in-cluster URL
                raw = r.read()
                return r.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            return e.code, None
    return send


class MlflowTracker:
    def __init__(self, uri: str, experiment: str, send: Sender | None = None) -> None:
        self.send = send or urllib_sender(uri)
        self.experiment = experiment
        self.run_id: str | None = None
        self.artifact_path: str | None = None

    def _call(self, method: str, path: str, body=None, raw: bytes | None = None,
              ctype: str = "application/json") -> dict:
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        status, out = self.send(method, path, data, ctype)
        if status >= 400:
            raise RuntimeError(f"MLflow {method} {path} -> HTTP {status}")
        return out or {}

    def _post(self, path: str, body: dict) -> dict:
        return self._call("POST", f"/api/2.0/mlflow/{path}", body)

    def _experiment_id(self) -> str:
        q = urllib.parse.urlencode({"experiment_name": self.experiment})
        status, out = self.send("GET", f"/api/2.0/mlflow/experiments/get-by-name?{q}", None, "application/json")
        if status == 200 and out:
            return out["experiment"]["experiment_id"]
        return self._post("experiments/create", {"name": self.experiment})["experiment_id"]

    def start(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        exp = self._experiment_id()
        info = self._post("runs/create", {"experiment_id": exp, "run_name": run_name, "start_time": _now_ms(),
                                          "tags": [{"key": k, "value": str(v)} for k, v in (tags or {}).items()]}
                          )["run"]["info"]
        self.run_id = info["run_id"]
        uri = info.get("artifact_uri", "")
        # with --serve-artifacts the run's artifacts live at mlflow-artifacts:/<exp>/<run>/artifacts
        self.artifact_path = uri.split("mlflow-artifacts:", 1)[1].lstrip("/") if "mlflow-artifacts:" in uri else None
        return self.run_id

    def log_params(self, params: dict) -> None:
        items = [{"key": k, "value": str(v)[:500]} for k, v in params.items()]
        for i in range(0, len(items), 100):              # API limit: 100 params per call
            self._post("runs/log-batch", {"run_id": self.run_id, "params": items[i:i + 100]})

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        ts = _now_ms()
        items = [{"key": k, "value": v, "timestamp": ts, "step": step} for k, v in metrics.items()]
        for i in range(0, len(items), 1000):             # API limit: 1000 metrics per call
            self._post("runs/log-batch", {"run_id": self.run_id, "metrics": items[i:i + 1000]})

    def log_json(self, name: str, obj) -> None:
        if not self.artifact_path:
            return
        self._call("PUT", f"/api/2.0/mlflow-artifacts/artifacts/{self.artifact_path}/{name}",
                   raw=json.dumps(obj, indent=2, default=str).encode())

    def finish(self, ok: bool = True) -> None:
        self._post("runs/update", {"run_id": self.run_id, "status": "FINISHED" if ok else "FAILED",
                                   "end_time": _now_ms()})


def log_training_run(uri: str | None, run_id: str, spec: dict, result: dict, extra: dict | None = None,
                     send: Sender | None = None) -> str | None:
    """One MLflow run per train-models run. Returns the MLflow run id, or None if tracking failed."""
    if not uri:
        return None
    try:
        t = MlflowTracker(uri, "intraday-models", send)
        t.start(run_id, {"verdict": result["verdict"], "barrier": result["barrier"]})
        t.log_params({**spec, "rows": result["rows"], "trainable_rows": result["trainable_rows"],
                      "cost_round_trip_pct": result["cost_round_trip_pct"],
                      "holdout_quarters": ",".join(result["holdout_quarters"])})
        # one step per fold so MLflow draws AUC over time
        quarters = sorted({m["quarter"] for m in result["fold_metrics"]})
        for step, q in enumerate(quarters):
            t.log_metrics({f"auc.{m['side']}.{m['model']}": m["auc"] for m in result["fold_metrics"]
                           if m["quarter"] == q and m.get("auc") is not None}, step=step)
        summary = {"chosen": result.get("chosen") or {}, "holdout": result.get("chosen_on_holdout") or {},
                   "base_win_rate": result["base_win_rate_after_costs"], **(extra or {})}
        t.log_metrics(flatten(summary))
        t.log_json("summary.json", {k: v for k, v in result.items() if not k.startswith("_")})
        t.finish(True)
        return t.run_id
    except Exception as exc:  # noqa: BLE001 - tracking must never break training
        log.warning("MLflow tracking failed (training still saved): %s", exc)
        return None
