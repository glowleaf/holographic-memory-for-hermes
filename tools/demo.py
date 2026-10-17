#!/usr/bin/env python
"""demo.py - show the holographic memory working, corpus by corpus, with real query output.

Runs a fixed battery against the live store: composition, entity probe, multi-entity reasoning,
cross-corpus keyword recall (notes / Hermes sessions / ChatGPT / Cursor), and contradiction checks.

  python demo.py [--db ...] [entity] [query]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

TOOLKIT = os.path.dirname(os.path.abspath(__file__))


def load_holo():
    spec = importlib.util.spec_from_file_location("holo", os.path.join(TOOLKIT, "holo.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["holo"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "memory_store.db"))
    ap.add_argument("--entity", default="Profile2")
    ap.add_argument("--query", default="stremio woke addon")
    a = ap.parse_args()

    holo = load_holo()
    st = holo.store_for(type("A", (), {"db": a.db, "trust": 0.5, "dim": 4096})())
    c = st._conn

    def rule(title):
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    rule("1. STORE COMPOSITION (what is in there)")
    total = c.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    print("facts: %d   entities: %d   entity links: %d   dim: 4096"
          % (total, c.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
             c.execute("SELECT COUNT(*) FROM fact_entities").fetchone()[0]))
    for label, like in (("Hermes chat sessions", "%session=%"), ("ChatGPT export", "%chatgpt=%"),
                        ("Cursor history", "%cursor=%"), ("notes/rules ingested", "%src=%")):
        n = c.execute("SELECT COUNT(*) FROM facts WHERE tags LIKE ?", (like,)).fetchone()[0]
        print("   %-24s %5d facts" % (label, n))
    for row in c.execute("SELECT category, COUNT(*) n FROM facts GROUP BY category ORDER BY n DESC"):
        print("   category %-12s %5d" % (row["category"], row["n"]))

    rule("2. ENTITY PROBE - every fact the store holds about %r (HRR unbind, ranked)" % a.entity)
    holo.print_ranked(holo.rank_by_vectors(st, [a.entity], "probe", None, 5))

    rule("3. MULTI-ENTITY REASON - facts where BOTH hold (vector-space AND, min over entities)")
    holo.print_ranked(holo.rank_by_vectors(st, ["Profile2", "Telegram"], "reason", None, 4))

    rule("4. KEYWORD RECALL - %r across all four corpora" % a.query)
    for r in holo.retriever_for(st, type("A", (), {"dim": 4096, "decay": 90})()).search(a.query, min_trust=0.0, limit=6):
        tags = r.get("tags", "")
        corpus = ("Hermes session" if "session=" in tags else "ChatGPT" if "chatgpt=" in tags
                  else "Cursor" if "cursor=" in tags else "notes")
        print("%.3f  [%-14s] %s" % (r["score"], corpus, " ".join(r["content"].split())[:130]))

    rule("5. CONTRADICTION SCAN (conflicting claims sharing a subject)")
    pairs = holo.retriever_for(st, type("A", (), {"dim": 4096, "decay": 90})()).contradict(threshold=0.35, limit=3)
    for p in pairs:
        print("%.2f  shared=%s" % (p["contradiction_score"], ",".join(p["shared_entities"])))
        print("     A: %s" % " ".join(p["fact_a"]["content"].split())[:110])
        print("     B: %s" % " ".join(p["fact_b"]["content"].split())[:110])
    print("   -> %d flagged (0 means no conflicting claims found)" % len(pairs))

    rule("6. QUANTIZED BANKS (superposition per period; SNR must stay >= 2.0)")
    rows = c.execute("SELECT bank_name, fact_count, dim FROM memory_banks ORDER BY fact_count DESC LIMIT 8").fetchall()
    for b in rows:
        n = int(b["fact_count"] or 0)
        snr = holo.HRR.snr_estimate(int(b["dim"]), n) if n else float("inf")
        print("   %-22s %5d items   SNR %s" % (b["bank_name"], n, ("%.1f" % snr) if snr != float("inf") else "inf"))
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
