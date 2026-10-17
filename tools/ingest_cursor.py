#!/usr/bin/env python
"""Ingest Cursor's local chat/composer history into the holographic fact store.

Cursor stores agent conversations in SQLite state DBs under
  %APPDATA%/Cursor/User/globalStorage/state.vscdb            (main: composerData, bubbleId, agentKv)
  %APPDATA%/Cursor/User/workspaceStorage/<hash>/state.vscdb  (per-workspace variants)
Relevant keys: composerData:<id> = conversation metadata, bubbleId:<composerId>:<bubbleId> = one
message bubble, agentKv:blob:<sha> = the raw message payloads (role/content JSON; some are binary).

Same two-tier output as the other ingesters: a header fact per conversation + capped content chunks.

  python ingest_cursor.py --max-chars 6000
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sqlite3
import sys
import time

TOOLKIT = os.path.dirname(os.path.abspath(__file__))
CURSOR_USER = os.path.expandvars(r"%APPDATA%/Cursor/User")


def load_holo():
    import importlib.util
    spec = importlib.util.spec_from_file_location("holo", os.path.join(TOOLKIT, "holo.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["holo"] = mod
    spec.loader.exec_module(mod)
    return mod


def dbs() -> list[str]:
    out = [p for p in [os.path.join(CURSOR_USER, "globalStorage", "state.vscdb")] if os.path.isfile(p)]
    out += sorted(glob.glob(os.path.join(CURSOR_USER, "workspaceStorage", "*", "state.vscdb")))
    return out


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    return str(value)


def bubble_text(payload: str) -> str:
    """A bubble row is JSON; pull the readable message text out of whichever field carries it."""
    try:
        obj = json.loads(payload)
    except ValueError:
        return ""
    if not isinstance(obj, dict):
        return ""
    for key in ("text", "content", "richText", "rawText"):
        val = obj.get(key)
        if isinstance(val, str) and len(val) >= 40:
            return " ".join(val.split())
    # some bubbles nest the message under 'message'
    inner = obj.get("message")
    if isinstance(inner, dict):
        return bubble_text(json.dumps(inner))
    return ""


def collect(db: str):
    """(composer_id, name, created_at, [ (order, role, text) ]) per conversation in one DB."""
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    con.text_factory = bytes
    try:
        meta = {}
        for key, value in con.execute("SELECT key, value FROM ItemTable WHERE key LIKE 'composerData%'"):
            k = as_text(key)
            if ":" not in k:
                continue
            cid = k.split(":", 1)[1]
            try:
                obj = json.loads(as_text(value))
            except ValueError:
                continue
            if isinstance(obj, dict):
                meta[cid] = obj
        try:
            for key, value in con.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"):
                k = as_text(key)
                cid = k.split(":", 1)[1]
                try:
                    obj = json.loads(as_text(value))
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    meta[cid] = obj
        except sqlite3.Error:
            pass
        bubbles: dict[str, list] = {}
        try:
            for key, value in con.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'bubbleId:%'"):
                k = as_text(key)
                parts = k.split(":")
                if len(parts) < 3:
                    continue
                text = bubble_text(as_text(value))
                if len(text) < 40:
                    continue
                try:
                    obj = json.loads(as_text(value))
                except ValueError:
                    obj = {}
                role = (obj.get("type") if isinstance(obj, dict) else None) or obj.get("role") or "?"
                order = obj.get("createdAt") or obj.get("timestamp") or 0 if isinstance(obj, dict) else 0
                bubbles.setdefault(parts[1], []).append((order, str(role), text))
        except sqlite3.Error:
            pass
        # fallback for DBs without bubbleId rows: raw agent payloads
        if not bubbles:
            for key, value in con.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'agentKv:blob:%'"):
                payload = as_text(value)
                if not payload.startswith("{"):
                    continue
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                role = obj.get("role")
                content = obj.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and len(content) >= 40:
                    bubbles.setdefault("agentKv", []).append((0, role, " ".join(content.split())))
        # Always harvest the raw agent message payloads too: only a subset of composers keep
        # bubbleId rows locally, and the rest would otherwise be metadata-only conversations.
        # Duplicates are impossible - facts.content is UNIQUE in the store.
        blobs = []
        try:
            for key, value in con.execute("SELECT key, value FROM cursorDiskKV WHERE key LIKE 'agentKv:blob:%'"):
                payload = as_text(value)
                if not payload.startswith("{"):
                    continue
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                role = obj.get("role")
                content = obj.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and len(content) >= 60:
                    text = " ".join(content.split())
                    if not text.startswith(("<user_info>", "<user_query>", "<system")):
                        blobs.append((0, role, text))
        except sqlite3.Error:
            pass
        if blobs:
            bubbles.setdefault("agentKv", []).extend(blobs)
        out = []
        for cid, msgs in bubbles.items():
            info = meta.get(cid, {})
            name = str(info.get("name") or info.get("title") or "")[:80]
            created = info.get("createdAt") or info.get("lastUpdatedAt") or 0
            out.append((cid, name, created, sorted(msgs, key=lambda x: x[0])))
        for cid, info in meta.items():
            if cid not in bubbles:
                out.append((cid, str(info.get("name") or "")[:80], info.get("createdAt") or 0, []))
        return out
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "memory_store.db"))
    ap.add_argument("--max-chars", dest="max_chars", type=int, default=6000)
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    holo = load_holo()
    st = holo.store_for(type("A", (), {"db": a.db, "trust": 0.5, "dim": 4096})())
    done = {r[0] for r in st._conn.execute(
        "SELECT DISTINCT substr(tags, instr(tags,'cursor=')+7, 40) FROM facts WHERE tags LIKE '%cursor=%'")}
    files = dbs()
    print("cursor state DBs: %d" % len(files))
    headers = chunks_added = 0
    seen = set()
    for db in files:
        for cid, name, created, msgs in collect(db):
            if cid in done or cid in seen:
                continue
            seen.add(cid)
            date = time.strftime("%Y-%m-%d", time.localtime((created or 0) / 1000 if created > 1e11 else (created or 0)))
            tags = "cursor;cursor=%s;date=%s;db=%s" % (cid, date, os.path.basename(os.path.dirname(db)))
            facts = []
            if msgs:
                first_user = next((t for _, r, t in msgs if r in ("user", "1", "human")), "")
                facts.append(("Cursor conversation \"%s\" (%s): %d messages. Opened with: %s"
                              % (name, date, len(msgs), first_user[:280]), tags + ";tier=header"))
            buf, used = "", 0
            for _, role, text in msgs:
                label = "user" if role in ("user", "1", "human") else "assistant"
                turn = "%s: %s" % (label, text)
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
            if not facts:
                continue
            if a.dry_run:
                headers += 1 if msgs else 0
                chunks_added += len([f for f in facts if "tier=content" in f[1]])
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
    if not a.dry_run:
        st._rebuild_bank("general")
    total = st._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    st.close()
    print("done: +%d headers, +%d content chunks from %d conversations | store now %d facts%s"
          % (headers, chunks_added, len(seen), total, " (DRY RUN)" if a.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
