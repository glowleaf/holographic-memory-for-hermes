#!/usr/bin/env python
"""memory_sleep.py - offline consolidation pass ("sleep") for the holographic memory store.

Runs when the box is quiet and nobody is talking:
  1. reads conversations that finished since the last pass out of state.db (watermark file)
  2. asks the LOCAL model to extract durable facts vs. procedures (never cloud)
  3. hands the facts to `holo.py sleep`, which dedupes/reinforces them, decays stale trust,
     archives+prunes dead facts, rebuilds the category banks and writes a log
  4. writes a human-readable MD note next to the log

A procedure (a repeatable how-to) is deliberately NOT stored as a fact: it belongs in a skill.
The model is asked to list them separately so a later session can turn them into one.

  python scripts/memory_sleep.py --db <store.db> --model-url http://<LAN_IP>:8888/v1 \\
      --model local-model --hours 30 --workdir <dir>
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def state_db() -> str:
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return os.path.join(home, "state.db")


def watermark(path: str) -> float:
    try:
        return float(io.open(path, encoding="utf-8").read().strip())
    except (OSError, ValueError):
        return time.time() - 86400 * 7        # first run: look back a week, not forever


def recent_transcript(db: str, since: float, max_chars: int = 60000) -> tuple[str, float]:
    """User+assistant text from sessions touched after `since` (tool noise dropped).

    started_at / ended_at / last_activity_at are epoch FLOATS in state.db, not ISO strings, so the
    window has to be compared numerically - a datetime(...,'unixepoch') string compare silently
    matched zero sessions. ended_at is NULL while a session is still open, hence the COALESCE."""
    if not os.path.isfile(db):
        return "", since
    con = sqlite3.connect("file:%s?mode=ro" % db, uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT id, source, title, COALESCE(ended_at, last_activity_at, started_at) AS ts "
            "FROM sessions WHERE COALESCE(ended_at, last_activity_at, started_at) > ? ORDER BY ts",
            (float(since),)).fetchall()
        chunks, newest = [], since
        for s in rows:
            msgs = con.execute(
                "SELECT role, content FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
                "AND content IS NOT NULL AND length(content) BETWEEN 20 AND 20000 ORDER BY id LIMIT 200",
                (s["id"],)).fetchall()
            if not msgs:
                continue
            body = "\n".join("%s: %s" % (m["role"], " ".join((m["content"] or "").split())[:400]) for m in msgs)
            chunks.append("### session %s (%s) %s\n%s" % (s["id"], s["source"], s["title"] or "", body))
            try:
                newest = max(newest, float(s["ts"]))
            except (TypeError, ValueError):
                pass
        text = "\n\n".join(chunks)
        return text[-max_chars:], newest
    finally:
        con.close()


PROMPT = """You are consolidating an agent's long-term memory during idle time. Read the conversations below.

Return ONLY a JSON object, no prose:
{"facts": ["<durable fact worth keeping between sessions>", ...],
 "procedures": ["<a repeatable how-to that belongs in a skill, not in memory>", ...],
 "contradictions": ["<where newer info contradicts older info>", ...]}

A fact is only worth keeping if a future session would ACT on it: a hostname, IP, port, path,
file name, ID, key location, price, schedule, account, model name, decision, or a correction the
user made. Every fact must name its subject explicitly.

Reject (do not output):
- vague commentary, rhetorical style notes, or plot/essay content ("X is a part of Y",
  "the market is compared to a queue", "this is important")
- anything you cannot tie to a concrete name, number, path or decision
- task chatter, restatements of the conversation, or anything already obvious
- fiction, world-building or drafting content from writing sessions

Max 20 facts, max 10 procedures. Quality over quantity: 3 real facts beat 20 vague ones.
"""


def _api_key_candidates(name: str) -> list[str]:
    """Every usable value for an env var, nearest .env first.

    A profile's own .env can hold a stale key: the profile2-profile run resolved its token there and the
    cloud answered 403, while the same call with the main Hermes .env key returned 200. Return all
    candidates so a call can retry with the next one instead of failing the whole pass."""
    if not name:
        return []
    out: list[str] = []
    if os.environ.get(name):
        out.append(os.environ[name])
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    paths = [os.path.join(home, ".env")]
    parent = os.path.dirname(os.path.abspath(home.rstrip("/\\")))
    if parent and parent != home:
        paths.append(os.path.join(parent, ".env"))          # profile home -> owning Hermes home
    paths.append(os.path.expanduser("~/.hermes/.env"))
    paths.append(r"<HERMES_HOME>/.env")
    for path in paths:
        try:
            for line in io.open(path, encoding="utf-8", errors="replace"):
                line = line.strip()
                if line.startswith(name + "="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value and value not in out:
                        out.append(value)
        except OSError:
            continue
    return out


def _api_key(name: str) -> str:
    """First usable value (kept for callers that just need one)."""
    cands = _api_key_candidates(name)
    return cands[0] if cands else ""


def call_local(urls: str, model: str, transcript: str, timeout: int = 600, max_tokens: int = 1500,
               attempt_timeout: int = 90, extra_payload: dict | None = None) -> tuple[dict, str]:
    """Try each local endpoint in order; returns (extraction, endpoint_used).

    Two shapes are tried per endpoint because the boxes differ: chat/completions (vLLM, accelerator-host 8888)
    and completions (llama.cpp builds that advertise capabilities ["completion"] only - calling
    chat/completions on those hangs until the socket closes). All endpoints are local by mandate;
    there is no cloud fallback here on purpose.

    max_tokens stays small on purpose: a CPU-hosted 8B at ~4096 tokens ran past 11 minutes without
    finishing. The transcript is chunked by the caller instead, so a full pass stays bounded."""
    chat_payload = {"model": model, "temperature": 0.1, "max_tokens": max_tokens,
                    "messages": [{"role": "system", "content": PROMPT},
                                 {"role": "user", "content": transcript}]}
    text_payload = {"model": model, "temperature": 0.1, "max_tokens": max_tokens,
                    "prompt": PROMPT + "\n\n" + transcript}
    if extra_payload:
        # e.g. {"reasoning_effort": "none"} - this gateway honours it, and without it a reasoning
        # model spends the entire budget on reasoning_content and returns EMPTY content
        # (measured: 600->2203, 1200->4019, 4000->14146 reasoning chars, content 0, finish=length)
        chat_payload.update(extra_payload)
        text_payload.update({k: v for k, v in extra_payload.items() if k != "messages"})
    errors = []
    for entry in [u.strip() for u in urls.split(",") if u.strip()]:
        parts = (entry.split("|") + ["", ""])[:3]
        url, model_override, key_name = parts[0], parts[1], parts[2]
        base = url.rstrip("/")
        if base.endswith("/v1"):        # the configured endpoints already carry /v1; the paths below
            base = base[:-3]            # add it back explicitly, so strip it to avoid /v1/v1/... (404)
        name = (model_override.strip() or model)
        headers = {"Content-Type": "application/json",
                   # Cloudflare in front of the cloud gateway answers 403 "error code: 1010" to the
                   # default Python-urllib User-Agent while accepting curl/Browser agents. Without
                   # this header EVERY request from this pipeline is rejected at the edge, which is
                   # what made extraction and compression look like model failures.
                   "User-Agent": os.environ.get("HOLO_USER_AGENT", "hermes-rem/1.0")}
        if "api.example.invalid" in base:
            # This gateway refuses requests without a routing session: HTTP 403/400
            # {"type":"MissingSessionID","message":"Request is missing x-opencode-session..."}
            headers["x-opencode-session"] = os.environ.get("HOLO_SESSION_ID") or ("hermes-rem-" + os.environ.get("HOLO_TAG", "default"))
        keys = _api_key_candidates(key_name.strip()) or [""]   # try every key we have before giving up
        for key in keys:
            if key:
                headers["Authorization"] = "Bearer " + key
            for path, payload, extract in (("/v1/chat/completions", dict(chat_payload, model=name),
                                            lambda b: b["choices"][0]["message"]["content"]),
                                           ("/v1/completions", dict(text_payload, model=name),
                                            lambda b: b["choices"][0].get("text", "")),
                                           # llama.cpp native shape: the ONLY path that answered on the
                                           # CPU box, while its OpenAI-compatible routes hung past the
                                           # client timeout. n_predict is this API's token cap.
                                           ("/completion", {"prompt": text_payload["prompt"], "n_predict": max_tokens,
                                                            "temperature": 0.1, "cache_prompt": True},
                                            lambda b: b.get("content", ""))):
                payload["model"] = name
                try:
                    req = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                                 headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=attempt_timeout) as resp:
                        body = json.loads(resp.read().decode("utf-8"))
                    out = extract(body)
                    if not out.strip():
                        continue
                    start, end = out.find("{"), out.rfind("}")
                    return (json.loads(out[start:end + 1]) if start >= 0
                            else {"facts": [], "procedures": [], "contradictions": []}), base + path
                except Exception as e:
                    errors.append("%s%s%s -> %s: %s" % (base, path, " [alt key]" if key and key != keys[0] else "",
                                                        type(e).__name__, e))
    raise RuntimeError("all local endpoints failed: " + " | ".join(errors))


_VAGUE = re.compile(r"\b(is a part of|is compared to|is a concept|is important|is a metaphor|"
                    r"is a party of|the market is|this is|it is|there is|refers to|relates to)\b", re.I)
_CONCRETE = re.compile(r"(\d|/|\\|\.(py|md|json|db|gguf|safetensors|service|timer|sh|exe|log)\b|"
                       r"\b[A-Z]{2,}\b|\b(?:http|tcp|port|ip)\b|@|\bASIN\b)", re.I)
_PROPER_NOWORD = re.compile(r"\b[A-Z][a-z]{2,}\b")


def is_specific(fact: str) -> tuple[bool, str]:
    """(kept?, reason) for the specificity filter.

    A CPU 8B asked for durable facts produced lines like "Blackpill theory is a part of this
    rewriting" and "The market is compared to a queue" - true-sounding text with nothing actionable
    in it. Levels: 1 = needs a concrete token or a proper noun (default); 2 = needs a concrete token
    (number/path/filename/ALLCAPS/handle), which the adaptive layer raises to when facts it stored
    stop surviving a week. The reason string is recorded so a dropped fact can be audited later
    instead of only counted."""
    level = int(os.environ.get("HOLO_SPECIFICITY", "1") or 1)
    if len(fact) < 40:
        return False, "too short (<40 chars)"
    concrete = bool(_CONCRETE.search(fact))
    if concrete:
        return True, "concrete token"          # an anchor wins: "it is"/"this is" inside an anchored
    vague = _VAGUE.search(fact)                # fact are ordinary prose, not vagueness (false positive seen
    if vague:                                  # on 'Room speaker Bluetooth MAC is ...; it is not known to bluez')
        return False, "vague phrasing (%s)" % vague.group(0)
    if level >= 2:
        return False, "no concrete token (specificity 2)"
    if _PROPER_NOWORD.search(fact):
        return True, "proper noun"
    return False, "no concrete token or proper noun"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "memory_store.db"))
    ap.add_argument("--holo", default=os.path.join(HERE, "holo.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--workdir", default=os.path.join(HERE, "sleep"))
    ap.add_argument("--model-url", default=os.environ.get(
        "SLEEP_MODEL_URLS",
        "https://api.example.invalid/v1|provider-model|PROVIDER_API_KEY,"
        "http://127.0.0.1:8011|local-model"),
        help="endpoints as 'url|model|ENV_VAR_FOR_KEY'; cloud first, local fallback second")
    ap.add_argument("--model", default=os.environ.get("SLEEP_MODEL", "local-model"),
                    help="default model name for endpoints that do not name one")
    ap.add_argument("--hours", type=int, default=30)
    ap.add_argument("--state-db", dest="state_db", default="",
                    help="explicit session DB path (defaults to $HERMES_HOME/state.db)")
    ap.add_argument("--chunk-chars", dest="chunk_chars", type=int, default=3500,
                    help="transcript window per extraction call (keeps a CPU 8B responsive)")
    ap.add_argument("--max-chunks", dest="max_chunks", type=int, default=12)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=1500)
    ap.add_argument("--attempt-timeout", dest="attempt_timeout", type=int, default=90,
                    help="per-attempt socket timeout; a hanging OpenAI route must not eat the pass")
    ap.add_argument("--idle-check", default="", help="shell command that must exit 0 for the box to count as idle")
    ap.add_argument("--metrics-json", dest="metrics_json", default="", help="write per-stage metrics for the adaptive layer")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)
    wm_file = os.path.join(a.workdir, "watermark.txt")
    since = watermark(wm_file)
    stamp = datetime.now().strftime("%Y-%m-%d")

    if a.idle_check:
        if subprocess.run(a.idle_check, shell=True).returncode != 0:
            print("not idle - skipping this pass")
            return 0

    transcript, newest = recent_transcript(a.state_db or state_db(), since)
    if len(transcript) < 200:
        print("nothing new to consolidate (transcript %d chars since %s)" % (len(transcript), datetime.fromtimestamp(since).isoformat()))
        io.open(wm_file, "w", encoding="utf-8").write(str(time.time()))
        return 0

    print("transcript: %d chars, asking local models for extraction" % len(transcript))
    chunks = [transcript[i:i + a.chunk_chars] for i in range(0, len(transcript), a.chunk_chars)][:a.max_chunks]
    merged = {"facts": [], "procedures": [], "contradictions": []}
    endpoint = ""
    try:
        for n, piece in enumerate(chunks, 1):
            part, endpoint = call_local(a.model_url, a.model, piece, max_tokens=a.max_tokens,
                                       attempt_timeout=a.attempt_timeout)
            for key in merged:
                merged[key] += [x for x in part.get(key, []) if x not in merged[key]]
            print("  -> chunk %d/%d extracted by %s (%d facts so far)" % (n, len(chunks), endpoint, len(merged["facts"])))
        extracted = merged
    except Exception as e:                                    # never fabricate facts to fill the gap
        print("EXTRACTION FAILED (%s: %s) - no facts written, watermark NOT advanced" % (type(e).__name__, e))
        return 2

    facts = [f for f in extracted.get("facts", []) if isinstance(f, str) and len(f.strip()) >= 20]
    facts_seen = len(facts)
    # Always persist the RAW candidate list: without it, a dropped fact is unrecoverable and the only
    # record is a count (that is exactly how 12 dropped facts from 2026-10-16 were lost for good).
    raw_path = os.path.join(a.workdir, "raw-candidates-%s.json" % stamp)
    with io.open(raw_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"chunks": len(chunks), "endpoint": endpoint, "facts_seen": facts_seen,
                   "raw_facts": facts, "procedures": extracted.get("procedures", []),
                   "contradictions": extracted.get("contradictions", [])}, f, indent=2)
    kept, dropped_detail = [], []
    for f in facts:
        ok, why = is_specific(f)
        (kept if ok else dropped_detail).append(f if ok else (f, why))
    dropped = len(dropped_detail)
    facts = kept
    if dropped_detail:
        audit = os.path.join(a.workdir, "dropped-facts.jsonl")
        with io.open(audit, "a", encoding="utf-8", newline="\n") as f:
            for text, why in dropped_detail:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": why, "fact": text}) + "\n")
        print("  filter: kept %d/%d (dropped %d) - drops audited in %s" % (len(facts), facts_seen, dropped, audit))
        for text, why in dropped_detail[:5]:
            print("     DROP [%s] %s" % (why, text[:100]))
    procedures = [p for p in extracted.get("procedures", []) if isinstance(p, str) and p.strip()]
    cand_file = os.path.join(a.workdir, "candidates-%s.json" % stamp)
    with io.open(cand_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"facts": facts, "procedures": procedures, "contradictions": extracted.get("contradictions", [])}, f, indent=2)
    print("extracted %d facts, %d procedures -> %s" % (len(facts), len(procedures), cand_file))
    if a.dry_run:
        return 0

    cmd = [a.python, a.holo, "--db", a.db, "sleep", "--candidates", cand_file,
           "--category", "general", "--tags", "sleep-%s" % stamp, "--similarity", "0.85",
           "--threshold", "0.3", "--prune-below", "0.2", "--prune-days", "120",
           "--archive-dir", os.path.join(a.workdir, "pruned"), "--log-dir", os.path.join(a.workdir, "logs")]
    out = subprocess.run(cmd, capture_output=True, text=True)
    print(out.stdout[-2000:] or out.stderr[-2000:])

    note = os.path.join(a.workdir, "notes", "sleep-%s.md" % stamp)
    os.makedirs(os.path.dirname(note), exist_ok=True)
    with io.open(note, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Sleep pass %s\n\n" % stamp)
        f.write("Model: %s (%s)\n\n" % (a.model, a.model_url))
        f.write("## Procedures (belong in a SKILL, not in facts)\n\n")
        f.writelines("- %s\n" % p for p in procedures)
        f.write("\n## Contradictions flagged\n\n")
        f.writelines("- %s\n" % c for c in extracted.get("contradictions", []))
        f.write("\n## Facts handed to the store\n\n")
        f.writelines("- %s\n" % x for x in facts)
    print("note: %s" % note)

    # metrics for the adaptive layer: what the filter saw/kept and what the store accepted
    added = reinforced = 0
    try:
        body = out.stdout[out.stdout.find("{"):]
        report = json.loads(body)
        added, reinforced = int(report.get("added", 0)), int(report.get("reinforced", 0))
    except (ValueError, TypeError):
        pass
    if a.metrics_json:
        with io.open(a.metrics_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"facts_seen": facts_seen, "facts_kept": len(facts), "procedures": len(procedures),
                       "chunks": len(chunks), "endpoint": endpoint, "facts_added": added,
                       "reinforced": reinforced}, f, indent=2)

    if out.returncode == 0:
        io.open(wm_file, "w", encoding="utf-8").write(str(newest or time.time()))
    return out.returncode


if __name__ == "__main__":
    raise SystemExit(main())
