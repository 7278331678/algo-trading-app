"""Nifty 100 universe: official NSE list, Kite tokens, sectors, point-in-time history.

Source of truth for the list is the NSE Indices constituents CSV. Membership is
kept as spells (valid_from, valid_to) so a backtest on any past date sees the
index as it stood then, including stocks that later dropped out.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOCAL_CSV = DATA_DIR / "ind_nifty100list.csv"           # optional manual override
HISTORY_CSV = DATA_DIR / "universe_history.csv"         # past membership spells (seed)

# NSE industry (as printed in the constituents CSV) -> sector index used for "is the sector green?"
# Names are Kite index tradingsymbols; any name Kite does not list falls back to NIFTY 50.
INDUSTRY_TO_SECTOR_INDEX: dict[str, str] = {
    "financial services": "NIFTY FIN SERVICE",
    "information technology": "NIFTY IT",
    "oil gas & consumable fuels": "NIFTY ENERGY",
    "power": "NIFTY ENERGY",
    "fast moving consumer goods": "NIFTY FMCG",
    "automobile and auto components": "NIFTY AUTO",
    "healthcare": "NIFTY PHARMA",
    "metals & mining": "NIFTY METAL",
    "realty": "NIFTY REALTY",
    "media entertainment & publication": "NIFTY MEDIA",
    "consumer durables": "NIFTY CONSR DURBL",
    "construction": "NIFTY INFRA",
    "construction materials": "NIFTY INFRA",
    "capital goods": "NIFTY INFRA",
    "telecommunication": "NIFTY INFRA",
}
BROAD_INDEX = "NIFTY 50"
CONTEXT_INDICES = ["NIFTY 50", "INDIA VIX", "NIFTY BANK"]


@dataclass(frozen=True)
class Constituent:
    symbol: str
    company_name: str
    industry: str
    isin: str
    series: str = "EQ"

    @property
    def sector_index(self) -> str:
        return INDUSTRY_TO_SECTOR_INDEX.get(self.industry.strip().lower(), BROAD_INDEX)


def parse_constituents_csv(text: str) -> list[Constituent]:
    """Parse the NSE Indices constituents CSV (Company Name, Industry, Symbol, Series, ISIN Code)."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames:
        raise ValueError("constituents CSV is empty")
    cols = {c.strip().lower(): c for c in reader.fieldnames}
    need = {"company name", "industry", "symbol", "isin code"}
    missing = need - cols.keys()
    if missing:
        raise ValueError(f"constituents CSV is missing columns: {sorted(missing)}")
    out: list[Constituent] = []
    for row in reader:
        sym = (row.get(cols["symbol"]) or "").strip().upper()
        if not sym:
            continue
        out.append(
            Constituent(
                symbol=sym,
                company_name=(row.get(cols["company name"]) or "").strip(),
                industry=(row.get(cols["industry"]) or "").strip(),
                isin=(row.get(cols["isin code"]) or "").strip(),
                series=(row.get(cols.get("series", ""), "EQ") or "EQ").strip(),
            )
        )
    if len({c.symbol for c in out}) != len(out):
        raise ValueError("constituents CSV has duplicate symbols")
    return out


def diff_membership(open_symbols: set[str], current: set[str]) -> tuple[set[str], set[str], set[str]]:
    """Returns (added, removed, kept) between stored open spells and today's list."""
    return current - open_symbols, open_symbols - current, open_symbols & current


def fetch_constituents_csv(url: str, timeout: float = 20.0) -> str:
    """Download the official list; fall back to a manually placed CSV if the site blocks us."""
    import httpx

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/csv,text/plain,*/*",
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        if "symbol" not in resp.text[:500].lower():
            raise ValueError("response does not look like the constituents CSV")
        return resp.text
    except Exception as exc:  # noqa: BLE001
        if LOCAL_CSV.exists():
            log.warning("download failed (%s); using %s", exc.__class__.__name__, LOCAL_CSV.name)
            return LOCAL_CSV.read_text(encoding="utf-8")
        raise RuntimeError(
            f"Could not download the Nifty 100 list ({exc.__class__.__name__}). Download "
            f"{url} in your browser and save it as backend/app/data/{LOCAL_CSV.name}, then rebuild."
        ) from exc


def load_history_seed() -> list[dict]:
    """Past membership spells: index_name,symbol,company_name,industry,valid_from,valid_to."""
    if not HISTORY_CSV.exists():
        return []
    rows = []
    for row in csv.DictReader(
        line for line in HISTORY_CSV.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ):
        if not (row.get("symbol") or "").strip():
            continue
        rows.append(
            {
                "index_name": row["index_name"].strip(),
                "symbol": row["symbol"].strip().upper(),
                "company_name": (row.get("company_name") or "").strip(),
                "industry": (row.get("industry") or "").strip(),
                "valid_from": date.fromisoformat(row["valid_from"].strip()),
                "valid_to": date.fromisoformat(row["valid_to"].strip()) if row.get("valid_to") else None,
            }
        )
    return rows


# ---------------------------------------------------------------- database side
def upsert_instruments(engine, kite_instruments: list[dict]) -> dict[tuple[str, str], int]:
    """Store NSE equities and indices; returns {(segment, tradingsymbol): token}."""
    from sqlalchemy import text

    keep = [
        i for i in kite_instruments
        if i.get("exchange") == "NSE"
        and (
            (i.get("segment") == "NSE" and i.get("instrument_type") == "EQ")
            or i.get("segment") == "INDICES"
        )
    ]
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO instruments (instrument_token, exchange_token, tradingsymbol, name, exchange,"
                " segment, instrument_type, tick_size, lot_size, updated_at)"
                " VALUES (:instrument_token, :exchange_token, :tradingsymbol, :name, :exchange, :segment,"
                " :instrument_type, :tick_size, :lot_size, now())"
                " ON CONFLICT (instrument_token) DO UPDATE SET tradingsymbol = EXCLUDED.tradingsymbol,"
                " name = EXCLUDED.name, tick_size = EXCLUDED.tick_size, lot_size = EXCLUDED.lot_size,"
                " updated_at = now()"
            ),
            [
                {
                    "instrument_token": int(i["instrument_token"]),
                    "exchange_token": int(i.get("exchange_token") or 0),
                    "tradingsymbol": i["tradingsymbol"],
                    "name": i.get("name"),
                    "exchange": i["exchange"],
                    "segment": i["segment"],
                    "instrument_type": i.get("instrument_type") or "",
                    "tick_size": i.get("tick_size"),
                    "lot_size": i.get("lot_size"),
                }
                for i in keep
            ],
        )
    return {(i["segment"], i["tradingsymbol"]): int(i["instrument_token"]) for i in keep}


def mark_fo_eligible(engine, nfo_instruments: list[dict]) -> int:
    """Flag NSE equities that have futures on NFO (they are under the Closing Auction Session)."""
    from sqlalchemy import text

    names = sorted({i["name"] for i in nfo_instruments if i.get("instrument_type") == "FUT" and i.get("name")})
    with engine.begin() as conn:
        conn.execute(text("UPDATE instruments SET fo_eligible = (tradingsymbol = ANY(:n))"
                          " WHERE segment = 'NSE' AND instrument_type = 'EQ'"), {"n": names})
        return conn.execute(text("SELECT count(*) FROM instruments WHERE fo_eligible")).scalar_one()


def sync_universe(engine, index_name: str, constituents: list[Constituent],
                  tokens: dict[tuple[str, str], int], today: date, baseline_from: date) -> dict:
    """Open/close membership spells so the table matches today's official list.

    On the very first sync, today's members are assumed to have been members since
    `baseline_from` (the start of the data window). That assumption is only removed by
    filling universe_history.csv with past index changes; stats["baseline_assumed"] flags it.
    """
    from sqlalchemy import text

    current = {c.symbol: c for c in constituents}
    indices_available = {sym for (seg, sym) in tokens if seg == "INDICES"}
    stats = {"added": [], "removed": [], "kept": 0, "missing_tokens": [], "sector_fallbacks": []}

    with engine.begin() as conn:
        # seed past spells once (source = 'seed'). A seed row WITHOUT valid_to is a current member
        # that joined on valid_from: it corrects the start of that member's open spell instead.
        seeds = load_history_seed()
        joined = {s["symbol"]: s["valid_from"] for s in seeds if s["index_name"] == index_name and not s["valid_to"]}
        for s in seeds:
            if not s["valid_to"]:
                continue
            exists = conn.execute(
                text("SELECT 1 FROM universe_members WHERE index_name=:i AND symbol=:s AND valid_from=:f"),
                {"i": s["index_name"], "s": s["symbol"], "f": s["valid_from"]},
            ).first()
            if not exists:
                conn.execute(
                    text(
                        "INSERT INTO universe_members (index_name, symbol, company_name, industry,"
                        " instrument_token, valid_from, valid_to, source)"
                        " VALUES (:i, :s, :n, :ind, :t, :f, :to, 'seed')"
                    ),
                    {"i": s["index_name"], "s": s["symbol"], "n": s["company_name"], "ind": s["industry"],
                     "t": tokens.get(("NSE", s["symbol"])), "f": s["valid_from"], "to": s["valid_to"]},
                )

        open_rows = {
            r[0] for r in conn.execute(
                text("SELECT symbol FROM universe_members WHERE index_name=:i AND valid_to IS NULL"),
                {"i": index_name},
            )
        }
        first_sync = not conn.execute(
            text("SELECT 1 FROM universe_members WHERE index_name=:i AND source <> 'seed' LIMIT 1"),
            {"i": index_name},
        ).first()
        start = baseline_from if first_sync else today
        stats["baseline_assumed"] = first_sync
        added, removed, kept = diff_membership(open_rows, set(current))
        for sym in sorted(removed):
            conn.execute(
                text("UPDATE universe_members SET valid_to=:d WHERE index_name=:i AND symbol=:s"
                     " AND valid_to IS NULL"),
                {"d": today - timedelta(days=1), "i": index_name, "s": sym},
            )
        for sym in sorted(added | kept):
            c = current[sym]
            sector = c.sector_index if c.sector_index in indices_available else BROAD_INDEX
            if sector != c.sector_index:
                stats["sector_fallbacks"].append(f"{sym}:{c.industry}")
            token = tokens.get(("NSE", sym))
            if token is None:
                stats["missing_tokens"].append(sym)
            params = {"i": index_name, "s": sym, "n": c.company_name, "ind": c.industry, "sec": sector,
                      "isin": c.isin, "t": token, "f": start}
            if sym in added:
                conn.execute(
                    text(
                        "INSERT INTO universe_members (index_name, symbol, company_name, industry, sector_index,"
                        " isin, instrument_token, valid_from) VALUES (:i, :s, :n, :ind, :sec, :isin, :t, :f)"
                    ),
                    params,
                )
            else:
                conn.execute(
                    text(
                        "UPDATE universe_members SET company_name=:n, industry=:ind, sector_index=:sec,"
                        " isin=:isin, instrument_token=:t WHERE index_name=:i AND symbol=:s AND valid_to IS NULL"
                    ),
                    params,
                )
        # current members that joined after the data start: their membership starts when they joined
        fixed = []
        for sym, since in joined.items():
            if sym in current:
                n = conn.execute(
                    text("UPDATE universe_members SET valid_from=:f WHERE index_name=:i AND symbol=:s"
                         " AND valid_to IS NULL AND valid_from <> :f"),
                    {"f": since, "i": index_name, "s": sym},
                ).rowcount
                if n:
                    fixed.append(sym)
        stats["join_dates_applied"] = sorted(fixed)
        stats["joined_but_not_current"] = sorted(set(joined) - set(current))
    stats.update(added=sorted(added), removed=sorted(removed), kept=len(kept))
    return stats


def members_on(engine, index_name: str, on: date) -> list[dict]:
    """Index members as of a date (point-in-time)."""
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT DISTINCT ON (symbol) symbol, company_name, industry, sector_index, instrument_token,"
                " valid_from, valid_to FROM universe_members WHERE index_name=:i AND valid_from <= :d"
                " AND (valid_to IS NULL OR valid_to >= :d) ORDER BY symbol, valid_from DESC"
            ),
            {"i": index_name, "d": on},
        ).mappings().all()
    return [dict(r) for r in rows]


def tokens_to_fetch(engine, index_name: str, since: date) -> dict[int, str]:
    """Every stock that was a member at any point since `since`, plus context and sector indices."""
    from sqlalchemy import text

    with engine.connect() as conn:
        stocks = conn.execute(
            text(
                "SELECT DISTINCT instrument_token, symbol FROM universe_members WHERE index_name=:i"
                " AND instrument_token IS NOT NULL AND (valid_to IS NULL OR valid_to >= :d)"
            ),
            {"i": index_name, "d": since},
        ).all()
        sectors = {r[0] for r in conn.execute(
            text("SELECT DISTINCT sector_index FROM universe_members WHERE sector_index IS NOT NULL")
        )}
        names = sorted(set(CONTEXT_INDICES) | sectors)
        idx = conn.execute(
            text("SELECT instrument_token, tradingsymbol FROM instruments WHERE segment='INDICES'"
                 " AND tradingsymbol = ANY(:names)"),
            {"names": names},
        ).all()
    return {int(t): s for t, s in [*stocks, *idx]}
