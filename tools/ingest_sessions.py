#!/usr/bin/env python
"""Ingest chat history out of state.db into the holographic fact store.

Two tiers, because the raw corpus is ~119M chars / 136k messages (~30M tokens): dumping that
verbatim would be ~99k chunks and a ~1.6GB vector table, and would bury the facts that matter.

  header  one fact per session: id, date, source, title, message count, opening user message.
          Cheap (906 facts) and makes every session discoverable as a unit.
  content user+assistant text of each session, capped per session (default 6000 chars, newest
          first), chunked on message boundaries into ~1200-char facts tagged with the session.

Resume-safe: a session already present (tag `session=<id>`) is skipped, and progress is appended
to a JSONL file so a long run can be interrupted and continued. Use --dry-run first.

  python ingest_sessions.py --db <store.db> --mode both --limit 20 --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sqlite3
import sys
import time

TOOLKIT = os.path.dirname(os.path.abspath(__file__))
GRAPH = re.compile(r"\b(session|state\.db|traceback|log|error|warning)\b", re.I)


def load_holo():
    import importlib.util
    spec = importlib.util.spec_from_file_location("holo", os.path.join(TOOLKIT, "holo.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["holo"] = mod
    spec.loader.exec_module(mod)
    return mod


DEEP_METRICS: dict = {}      # filled by deep_pass, written when --metrics-json is set


def session_rows(state_db: str, limit: int | None, newest_first: bool = True, sources=None):
    con = sqlite3.connect("file:%s?mode=ro" % state_db, uri=True)
    con.row_factory = sqlite3.Row
    order = "DESC" if newest_first else "ASC"
    sql = ("SELECT id, source, title, started_at, message_count FROM sessions "
           "WHERE message_count > 1 ORDER BY COALESCE(last_activity_at, started_at) " + order)
    rows = [dict(r) for r in con.execute(sql)]
    con.close()
    if sources:
        keep = set(sources.split(","))
        rows = [r for r in rows if r["source"] in keep]
    return rows[:limit] if limit else rows


def session_messages(state_db: str, sid: str, max_chars: int, per_msg: int = 1200):
    con = sqlite3.connect("file:%s?mode=ro" % state_db, uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
        "AND content IS NOT NULL AND length(content) >= 40 ORDER BY id", (sid,)).fetchall()
    con.close()
    out, used = [], 0
    for r in rows:
        text = " ".join((r["content"] or "").split())
        if not text or GRAPH.search(text[:120]):          # skip log/traceback dumps
            continue
        tag = "user" if r["role"] == "user" else "assistant"
        piece = "%s: %s" % (tag, text[:per_msg])
        if used + len(piece) > max_chars:
            break
        out.append(piece)
        used += len(piece)
    return out


def deep_pass(holo, st, a) -> int:
    """Full-text pass over the OLDEST sessions that only have a capped header so far.

    The first ingest capped every session at 6,000 chars; this walks the backlog at --deep-chars per
    session, oldest first, and tags the result tier=deep so a re-run skips it. Designed to run a
    bounded slice per night (downtime work) instead of one 119M-char dump."""
    con = sqlite3.connect("file:%s?mode=ro" % a.state_db, uri=True)
    con.row_factory = sqlite3.Row
    sessions = [dict(r) for r in con.execute(
        "SELECT id, source, title, started_at, message_count FROM sessions WHERE message_count > 1")]
    con.close()
    deep_done = {r[0] for r in st._conn.execute(
        "SELECT DISTINCT substr(tags, instr(tags,'session=')+8, 20) FROM facts WHERE tags LIKE '%tier=deep%'")}
    sessions = [s for s in sessions if s["id"][:20] not in deep_done]
    sessions.sort(key=lambda s: s["started_at"] or 0)          # oldest first
    todo = sessions[:a.deep]
    DEEP_METRICS.update({"pending": len(sessions), "done": len(todo)})
    print("deep pass: %d sessions pending, doing %d oldest (cap %d chars each)"
          % (len(sessions), len(todo), a.deep_chars))
    added = 0
    t0 = time.time()
    for n, s in enumerate(todo, 1):
        sid = s["id"]
        msgs = session_messages(a.state_db, sid, a.deep_chars)
        if not msgs:
            continue
        date = time.strftime("%Y-%m-%d", time.localtime(s["started_at"] or 0))
        tags = "chat;session=%s;src=%s;date=%s;tier=deep" % (sid, s["source"], date)
        buf = ""
        facts = []
        for m in msgs:
            if len(buf) + len(m) + 2 <= a.chunk:
                buf = (buf + " " + m).strip()
            else:
                if buf:
                    facts.append(buf)
                buf = m[:a.chunk]
        if buf:
            facts.append(buf)
        if a.dry_run:
            added += len(facts)
            print("   %s -> %d chunks" % (sid, len(facts)))
            continue
        for content in facts:
            vocab, counts = holo.build_vocab([content])
            try:
                holo.insert_fact_with_entities(st, content, "general", tags,
                                               holo.entities_in(content, vocab, counts, 3))
                added += 1
            except sqlite3.IntegrityError:
                pass
        if n % 10 == 0 or n == len(todo):
            st._rebuild_bank("general")
            print("  %d/%d sessions | +%d chunks | %.0fs" % (n, len(todo), added, time.time() - t0), flush=True)
    if not a.dry_run:
        st._rebuild_bank("general")
    total = st._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    st.close()
    DEEP_METRICS["chunks"] = added
    print("deep pass done: +%d chunks from %d sessions | store now %d facts%s"
          % (added, len(todo), total, " (DRY RUN)" if a.dry_run else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "memory_store.db"))
    ap.add_argument("--state-db", default=os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state.db"))
    ap.add_argument("--mode", choices=("header", "content", "both"), default="both")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-chars", dest="max_chars", type=int, default=6000, help="per session content budget")
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--sources", default="", help="comma-separated source filter (desktop,telegram,cli)")
    ap.add_argument("--progress", default=os.path.join(TOOLKIT, "sleep", "session-ingest.jsonl"))
    ap.add_argument("--deep", type=int, default=0,
                    help="deep-history mode: fully ingest the N OLDEST sessions that only have a header")
    ap.add_argument("--deep-chars", dest="deep_chars", type=int, default=40000)
    ap.add_argument("--metrics-json", dest="metrics_json", default="", help="write stage metrics for the adaptive layer")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    holo = load_holo()
    st = holo.store_for(type("A", (), {"db": a.db, "trust": 0.5, "dim": 4096})())

    if a.deep:
        code = deep_pass(holo, st, a)
        if a.metrics_json:
            try:
                with io.open(a.metrics_json, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(DEEP_METRICS, f, indent=2)
            except OSError:
                pass
        return code

    existing = {r[0] for r in st._conn.execute(
        "SELECT DISTINCT substr(tags, instr(tags, 'session=') + 8, 20) FROM facts WHERE tags LIKE '%session=%'")}
    rows = session_rows(a.state_db, a.limit or None, sources=a.sources or None)
    print("sessions to consider: %d (already in store: %d)" % (len(rows), len(existing)))

    os.makedirs(os.path.dirname(a.progress), exist_ok=True)
    vocab = holo.build_vocab([r["content"] for r in st._conn.execute("SELECT content FROM facts LIMIT 5000")])
    headers = contents = skipped = 0
    t0 = time.time()
    for i, s in enumerate(rows, 1):
        sid = s["id"]
        date = time.strftime("%Y-%m-%d", time.localtime(s["started_at"] or 0))
        tag_base = "chat;session=%s;src=%s;date=%s" % (sid, s["source"], date)
        if sid in existing or any(sid[:20] in e for e in existing):
            skipped += 1
            continue
        msgs = session_messages(a.state_db, sid, a.max_chars)
        opening = next((m[6:] for m in msgs if m.startswith("user:")), "")
        facts = []
        if a.mode in ("header", "both"):
            facts.append(("Session %s (%s, %s) \"%s\": %d messages. Opened with: %s"
                          % (sid, date, s["source"], (s["title"] or "")[:80], s["message_count"],
                             " ".join(opening.split())[:300]), tag_base + ";tier=header"))
        if a.mode in ("content", "both") and msgs:
            buf = ""
            for m in msgs:
                if len(buf) + len(m) + 2 <= a.chunk:
                    buf = (buf + " " + m).strip()
                else:
                    if buf:
                        facts.append((buf, tag_base + ";tier=content"))
                    buf = m[:a.chunk]
            if buf:
                facts.append((buf, tag_base + ";tier=content"))
        if a.dry_run:
            headers += 1 if a.mode in ("header", "both") else 0
            contents += len([f for f in facts if "tier=content" in f[1]])
            if i <= 3:
                for f, t in facts[:2]:
                    print("   [%s] %s" % (t.split(";")[-1], f[:110]))
            continue
        for content, tags in facts:
            ents = holo.entities_in(content, *holo.build_vocab([content]), 3)
            try:
                holo.insert_fact_with_entities(st, content, "general", tags, ents)
                if "tier=content" in tags:
                    contents += 1
                else:
                    headers += 1
            except sqlite3.IntegrityError:
                pass
        with io.open(a.progress, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"session": sid, "facts": len(facts), "date": date}) + "\n")
        if i % 25 == 0 or i == len(rows):
            st._rebuild_bank("general")
            print("  %d/%d sessions | +%d headers, +%d content chunks | %.0fs"
                  % (i, len(rows), headers, contents, time.time() - t0), flush=True)

    if not a.dry_run:
        st._rebuild_bank("general")
    total = st._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    st.close()
    print("done: +%d headers, +%d content chunks, %d sessions skipped | store now %d facts%s"
          % (headers, contents, skipped, total, " (DRY RUN, nothing written)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
