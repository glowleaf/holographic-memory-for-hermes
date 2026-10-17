#!/usr/bin/env python
"""Ingest the ChatGPT export (conversations.json) into the holographic fact store.

Same two-tier shape as ingest_sessions.py: a header fact per conversation plus its text capped per
conversation, chunked into atomic facts. The export here is 828 conversations / ~101M chars, so the
cap (not the whole transcript) is what keeps the store usable; raise --max-chars for depth.

  python ingest_chatgpt.py --file "<WORKSPACE>/export/conversations.json"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import sys
import time

TOOLKIT = os.path.dirname(os.path.abspath(__file__))


def load_holo():
    import importlib.util
    spec = importlib.util.spec_from_file_location("holo", os.path.join(TOOLKIT, "holo.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["holo"] = mod
    spec.loader.exec_module(mod)
    return mod


def message_text(msg: dict) -> str:
    """Plain text of a ChatGPT message node (content.parts is a list of str/dicts)."""
    content = (msg or {}).get("content") or {}
    parts = content.get("parts") or []
    out = []
    for part in parts:
        if isinstance(part, str):
            out.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            out.append(part["text"])
    text = " ".join(" ".join(out).split())
    return text


def conversation_turns(conv: dict) -> tuple[list[str], dict]:
    """Ordered user/assistant turns + metadata. mapping is a tree keyed by node id."""
    nodes = (conv.get("mapping") or {}).values()
    msgs = []
    for node in nodes:
        msg = node.get("message") or {}
        role = ((msg.get("author") or {}).get("role")) or ""
        if role not in ("user", "assistant"):
            continue
        text = message_text(msg)
        if len(text) < 40:
            continue
        ts = msg.get("create_time") or 0
        msgs.append((ts, role, text))
    msgs.sort(key=lambda x: x[0] or 0)
    meta = {"title": conv.get("title") or "", "create_time": conv.get("create_time") or (msgs[0][0] if msgs else 0),
            "update_time": conv.get("update_time") or 0, "turns": len(msgs),
            "user_turns": sum(1 for m in msgs if m[1] == "user")}
    return ["%s: %s" % (r, t) for _, r, t in msgs], meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True,
                    help="path to conversations.json from your chat export")
    ap.add_argument("--db", default=os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "memory_store.db"))
    ap.add_argument("--max-chars", dest="max_chars", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    holo = load_holo()
    st = holo.store_for(type("A", (), {"db": a.db, "trust": 0.5, "dim": 4096})())
    done = {r[0] for r in st._conn.execute(
        "SELECT DISTINCT substr(tags, instr(tags,'chatgpt=')+8, 40) FROM facts WHERE tags LIKE '%chatgpt=%'")}

    print("reading %s (this is a large file)..." % a.file, flush=True)
    convs = json.load(io.open(a.file, encoding="utf-8", errors="replace"))
    print("conversations: %d (already in store: %d)" % (len(convs), len(done)))
    headers = chunks_added = skipped = 0
    t0 = time.time()
    for i, conv in enumerate(convs, 1):
        cid = conv.get("conversation_id") or ""
        if cid in done:
            skipped += 1
            continue
        turns, meta = conversation_turns(conv)
        date = time.strftime("%Y-%m-%d", time.localtime(meta["create_time"] or 0))
        tags = "chatgpt;chatgpt=%s;date=%s" % (cid, date)
        facts = [("ChatGPT conversation \"%s\" (%s): %d turns (%d from him). Opened with: %s"
                  % ((meta["title"] or "")[:90], date, meta["turns"], meta["user_turns"],
                     next((t[6:] for t in turns if t.startswith("user:")), "")[:280]),
                  tags + ";tier=header")]
        buf, used = "", 0
        for turn in turns:
            if used + len(turn) > a.max_chars:
                break
            if len(buf) + len(turn) + 2 <= a.chunk:
                buf = (buf + " " + turn).strip()
            else:
                if buf:
                    facts.append((buf, tags + ";tier=content"))
                buf = turn[:a.chunk]
            used += len(turn)
        if buf:
            facts.append((buf, tags + ";tier=content"))
        if a.dry_run:
            headers += 1
            chunks_added += len(facts) - 1
            if i <= 2:
                for f, t in facts[:3]:
                    print("   [%s] %s" % (t.split(";")[-1], f[:110]))
            continue
        for content, ftags in facts:
            vocab, counts = holo.build_vocab([content])
            try:
                holo.insert_fact_with_entities(st, content, "general", ftags, holo.entities_in(content, vocab, counts, 3))
                if "tier=content" in ftags:
                    chunks_added += 1
                else:
                    headers += 1
            except sqlite3.IntegrityError:
                pass
        if i % 50 == 0 or i == len(convs):
            st._rebuild_bank("general")
            print("  %d/%d | +%d headers, +%d chunks | %.0fs" % (i, len(convs), headers, chunks_added, time.time() - t0), flush=True)
        if a.limit and i >= a.limit:
            break
    if not a.dry_run:
        st._rebuild_bank("general")
    total = st._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    st.close()
    print("done: +%d headers, +%d content chunks, %d already present | store now %d facts%s"
          % (headers, chunks_added, skipped, total, " (DRY RUN)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
