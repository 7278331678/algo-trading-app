import json
from datetime import date, time

import pytest

from app.core.sessions import SessionRules


def test_bundled_rules():
    r = SessionRules()
    assert r.continuous_end(date(2026, 7, 31), cas=True) == time(15, 30)
    assert r.continuous_end(date(2026, 11, 27), cas=True) == time(15, 15)
    assert r.continuous_end(date(2026, 11, 27), cas=False) == time(15, 30)
    assert r.square_off(date(2026, 11, 27), cas=True) == time(15, 0)
    assert not r.period(date(2026, 12, 7)).confirmed  # must be confirmed before live trading


def test_rejects_square_off_after_broker(tmp_path):
    bad = {"periods": [{"from": "2026-01-01", "label": "x",
                        "continuous_end": {"cas": "15:15", "non_cas": "15:30"},
                        "broker_mis_square_off": {"cas": "15:12", "non_cas": "15:25"},
                        "our_square_off": {"cas": "15:13", "non_cas": "15:15"}}]}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        SessionRules(p)
