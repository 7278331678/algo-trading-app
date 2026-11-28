"""Rule-based categories for exchange announcements (no LLM: fast, deterministic, testable).

NSE's own subject line ("desc") is already a coarse category; we map it, plus the summary
text, onto the few kinds of news that plausibly move a stock intraday. Routine filings
(trading-window closures, newspaper copies, compliance certificates) are kept apart so they
don't drown out real news in the counts.
"""

from __future__ import annotations

import re

# checked in order: the first match wins
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("clarification", ("clarification", "news verification", "spurt in volume", "price movement",
                       "movement in price")),
    ("results", ("financial result", "quarterly result", "audited result", "unaudited result",
                 "results for the quarter", "integrated filing (financial)")),
    ("order", ("award of order", "receipt of order", "bagging", "receiving of order", "order win",
               "letter of award", "contract award", "work order")),
    ("rating", ("credit rating", "rating action", "rating reaffirm", "rating upgrade", "rating downgrade")),
    ("mna", ("acquisition", "acquires", "acquired", "amalgamation", "merger", "demerger", "scheme of arrangement",
             "divest", "sale or disposal", "disposal of", "slump sale", "joint venture", "open offer",
             "restructuring")),
    ("corp_action", ("dividend", "bonus", "buyback", "buy back", "sub-division", "split", "rights issue",
                     "fund raising", "qualified institutions placement", "qip", "preferential issue")),
    ("holder", ("takeover", "sast", "pledge", "encumbrance", "insider trading", "promoter",
                "substantial acquisition", "block deal", "bulk deal")),
    ("regulatory", ("action(s) taken", "orders passed", "show cause", "penalty", "sebi order", "litigation",
                    "dispute", "search and seizure", "raid", "income tax", "gst", "enforcement", "fraud",
                    "default", "insolvency", "nclt")),
    ("kmp", ("change in director", "key managerial", "resignation", "appointment of", "cessation",
             "managing director", "chief executive", "chief financial")),
    ("guidance", ("press release", "investor presentation", "business update", "operational update",
                  "sales update", "monthly sales", "production update", "provisional", "media release")),
    ("routine", ("trading window", "newspaper", "investor meet", "analyst", "con. call", "conference call",
                 "earnings call", "compliance certificate", "certificate under", "regulation 74",
                 "loss of share", "duplicate share", "esop", "esos", "allotment", "shareholders meeting",
                 "annual general meeting", "agm", "postal ballot", "record date", "book closure",
                 "board meeting intimation", "intimation of board meeting", "updates", "general")),
]

CATEGORIES = [c for c, _ in RULES] + ["other"]
MATERIAL = ["results", "order", "rating", "mna", "corp_action", "holder", "regulatory", "kmp",
            "guidance", "clarification"]

_compiled = [(cat, re.compile("|".join(re.escape(k) for k in keys))) for cat, keys in RULES]


# subjects that say nothing: the text decides (but never "results": transcripts and notices mention results)
GENERIC_SUBJECTS = {"updates", "general updates", "general", "others", "other", "disclosure"}
BODY_CAN_SAY = {"order", "rating", "mna", "corp_action", "regulatory", "clarification", "kmp", "guidance"}


def classify(subject: str, body: str | None = None) -> str:
    """The subject is the exchange's own label and decides; the text only refines generic or unknown subjects."""
    s = (subject or "").lower().strip()
    for cat, rx in _compiled:
        if rx.search(s):
            if cat == "routine" and s in GENERIC_SUBJECTS:
                return _first((body or "").lower(), allowed=BODY_CAN_SAY) or "routine"
            return cat
    # unknown subject (e.g. "Outcome of Board Meeting"): the text decides, results included
    return _first((body or "").lower(), allowed=set(CATEGORIES) - {"routine"}) or "other"


def _first(text: str, allowed: set[str]) -> str | None:
    for cat, rx in _compiled:
        if cat in allowed and rx.search(text):
            return cat
    return None


def is_results_meeting(purpose: str) -> bool:
    p = (purpose or "").lower()
    return any(k in p for k in ("financial result", "results", "audited", "unaudited", "quarterly"))
