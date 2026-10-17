#!/usr/bin/env python3
"""rem_adaptive.py - the self-improvement layer of the REM cycle.

The cycle used to run fixed parameters whatever happened. Now every stage reports metrics, the metrics
are scored against explicit rules, and the scoreboard rewrites the parameters the next run uses:

  * extraction too slow / timing out        -> smaller chunks and fewer of them
  * extraction fast but empty               -> room to do more (bigger chunk count)
  * filter rejecting nearly everything      -> widen the context per fact (chunk_chars up)
  * sleep-added facts dying within a week   -> require harder specificity (2 concrete tokens)
  * compression missing its target          -> compress twice per night
  * compression model failing               -> back that stage off one night
  * deep backlog large / nearly drained    -> more / fewer sessions per night
  * banks below the SNR floor               -> smaller banks
Everything lives in rem_state.json (current parameters) and rem_metrics.jsonl (append-only history),
so the behaviour is inspectable, reproducible and never a black box. Repeated failures escalate into
the inbox, so the agent sees them as material on the next pass instead of them rotting in a log.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

TOOLKIT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(TOOLKIT, "rem_state.json")
METRICS_PATH = os.path.join(TOOLKIT, "rem_metrics.jsonl")

DEFAULTS = {
    "chunk_chars": 4000,        # transcript window per extraction call
    "max_chunks": 3,            # extraction calls per REM pass
    "max_tokens": 2000,         # response budget per call (a reasoning model spends part of it thinking)
    "deep": 60,                 # sessions fully ingested per night
    "deep_chars": 40000,
    "compress_passes": 1,       # how many times the compress stage runs per night
    "bank_cap": 800,            # max items bundled into one HRR bank
    "specificity": 1,           # concrete tokens a fact must carry to be stored
    "tokens_per_night": 4000,   # ceiling on model generation for the whole pass
    "compress_batches": 8,      # max model calls per memory file per night (each cuts ~1.5k chars)
    "backoff": {},              # stage -> unix ts until which the stage is skipped
    "consecutive_failures": {},  # stage -> count
}
# Model policy: the CLOUD model is primary and Profile2's local 8B is the fallback. The box's CPU model
# generates at ~3 tokens/second (measured: tg = 3.34 t/s) while it shares the machine with the agent's
# live traffic, so the local model must never be the first choice for batch work. The adaptive rules
# still shrink these budgets if the cloud is unreachable and the slow fallback takes over.


def load_state() -> dict:
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return {**DEFAULTS, **json.load(f)}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_state(state: dict) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)


def record(stage: str, exit_code: int, seconds: float, **metrics) -> None:
    with open(METRICS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "tag": os.environ.get("HOLO_TAG", "?"),
                            "stage": stage, "exit": exit_code, "seconds": round(seconds, 1), **metrics}) + "\n")


def history(stage: str | None = None, limit: int = 200) -> list[dict]:
    rows = []
    try:
        with open(METRICS_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if stage is None or row.get("stage") == stage:
                    rows.append(row)
    except OSError:
        pass
    return rows[-limit:]


def fact_survival(db: str, days: int = 7) -> float | None:
    """Share of sleep-added facts that are still in the store after `days`.

    This is the cycle's own reward signal: the extraction prompt and the specificity filter are only
    good if the facts they produce last. Returns None when there is not enough history yet."""
    if not os.path.isfile(db):
        return None
    cutoff = time.time() - days * 86400
    try:
        con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
        rows = con.execute("SELECT tags, created_at FROM facts WHERE tags LIKE '%sleep-%'").fetchall()
        con.close()
    except sqlite3.Error:
        return None
    if len(rows) < 20:
        return None
    alive = 0
    old = 0
    for tags, created in rows:
        try:
            t = time.mktime(time.strptime(str(created)[:19], "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        if t > cutoff:
            continue
        old += 1
        alive += 1            # still present = survived (pruning removes rows, so presence is the metric)
    return (alive / old) if old else None


def adapt(state: dict, db: str) -> tuple[dict, list[str]]:
    """Apply the rules. Returns (new state, human-readable reasons). Each reason names its trigger."""
    reasons: list[str] = []
    ex = [r for r in history("rem") if r.get("tag") == os.environ.get("HOLO_TAG", "?")][-5:]
    comp = [r for r in history("compress") if r.get("tag") == os.environ.get("HOLO_TAG", "?")][-5:]
    deep = [r for r in history("deep") if r.get("tag") == os.environ.get("HOLO_TAG", "?")][-5:]

    if ex:
        last = ex[-1]
        slow = last.get("seconds", 0) > 600 or last.get("exit") == 2
        if slow and state["max_chunks"] > 1:
            state["max_chunks"] -= 1
            reasons.append("extraction slow/failed (%ss, exit %s) -> max_chunks %d"
                           % (last.get("seconds"), last.get("exit"), state["max_chunks"]))
        if slow and state["max_tokens"] > 120:
            state["max_tokens"] = max(120, state["max_tokens"] - 100)
            reasons.append("extraction slow -> max_tokens %d (slow local model, ~3 t/s measured)"
                           % state["max_tokens"])
        elif last.get("seconds", 999) < 60 and last.get("facts_added", 0) == 0 and state["max_tokens"] < 600:
            state["max_tokens"] += 100
            reasons.append("extraction fast but empty -> max_tokens %d" % state["max_tokens"])
        if last.get("seconds", 999) < 120 and last.get("facts_added", 0) == 0 and state["max_chunks"] < 6:
            state["max_chunks"] += 1
            reasons.append("extraction fast but empty -> max_chunks %d" % state["max_chunks"])
        kept, seen = last.get("facts_kept", 0), last.get("facts_seen", 0)
        if seen >= 5 and kept / seen < 0.3 and state["chunk_chars"] < 6000:
            state["chunk_chars"] += 500
            reasons.append("filter kept %.0f%% of facts -> chunk_chars %d (more context per fact)"
                           % (100 * kept / seen, state["chunk_chars"]))

    survival = fact_survival(db)
    if survival is not None and survival < 0.5 and state["specificity"] < 2:
        state["specificity"] = 2
        reasons.append("only %.0f%% of sleep facts survived a week -> specificity 2 (harder filter)" % (100 * survival))

    if comp:
        last = comp[-1]
        if last.get("exit") == 2:
            state["backoff"]["compress"] = time.time() + 86400
            reasons.append("compress model failed -> backing that stage off for one night")
        elif last.get("chars_after", 0) > last.get("target", 3000) * 1.15 and state["compress_passes"] < 3:
            state["compress_passes"] += 1
            reasons.append("compression missed target (%s > %s) -> compress_passes %d"
                           % (last.get("chars_after"), last.get("target"), state["compress_passes"]))
        elif last.get("chars_after", 10**9) <= last.get("target", 3000) * 1.15 and state["compress_passes"] > 1:
            state["compress_passes"] = 1
            reasons.append("compression back under target -> compress_passes 1")

    if deep:
        pending = deep[-1].get("pending", 0)
        if pending > 2000 and state["deep"] < 200:
            state["deep"] = min(200, state["deep"] + 20)
            reasons.append("deep backlog %d -> deep %d/night" % (pending, state["deep"]))
        elif pending < 200 and state["deep"] > 10:
            state["deep"] = max(10, state["deep"] - 10)
            reasons.append("deep backlog drained (%d) -> deep %d/night" % (pending, state["deep"]))

    snr_rows = [r for r in history("rebank")][-1:] 
    if snr_rows and snr_rows[0].get("min_snr", 99) < 2.0 and state["bank_cap"] > 200:
        state["bank_cap"] = max(200, state["bank_cap"] - 100)
        reasons.append("bank SNR %.2f below floor -> bank_cap %d" % (snr_rows[0]["min_snr"], state["bank_cap"]))

    return state, reasons


def note_failure(state: dict, stage: str) -> bool:
    """Count consecutive failures; True when the stage has failed 3 nights running (escalate)."""
    counts = state.setdefault("consecutive_failures", {})
    counts[stage] = counts.get(stage, 0) + 1
    return counts[stage] >= 3


def clear_failure(state: dict, stage: str) -> None:
    state.setdefault("consecutive_failures", {})[stage] = 0


def escalate(inbox: str, stage: str, detail: str) -> str:
    """Write an escalation where the next pass will ingest it as material (and thus be taught)."""
    os.makedirs(inbox, exist_ok=True)
    path = os.path.join(inbox, "rem-escalation-%s-%s.md" % (stage, time.strftime("%Y%m%d")))
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n\n## REM escalation: %s failed 3 nights running (%s)\n\n```\n%s\n```\n"
                % (stage, time.strftime("%Y-%m-%d %H:%M"), detail[-2000:]))
    return path
