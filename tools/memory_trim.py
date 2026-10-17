#!/usr/bin/env python3
"""memory_trim.py - the COMPRESS stage of the REM cycle.

Three passes, cheapest first, so a small model is only asked to do what it is good at:

  A. RULE PASS (no model): dated/historical entries and log junk leave the always-on file and are
     APPENDED to knowledge files (topic from keywords). This is what actually shrinks a 50k file.
  B. MODEL PASS: the surviving rule-like entries are tightened, in <=BATCH-char groups so the request
     always fits a small local context window (the box's llama-server runs --ctx-size 8192, and a
     40k-char prompt is an HTTP 400, not a slow answer).
  C. VERIFY: result must land under target*1.15, else the original file is restored from the archive.

Nothing is ever lost: the full pre-REM file is copied to an archive dir first, and every moved entry
is written to a knowledge file before the always-on file is rewritten.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import sys
import time

TOOLKIT = os.path.dirname(os.path.abspath(__file__))
DELIM = "\n\u00a7\n"

# A. rule-based classification
_DATED = re.compile(r"\b(20\d\d-\d\d-\d\d|20\d\d-\d\d)\b")
_HISTORY_WORDS = re.compile(r"\b(maintenance pass|verified (?:on|live)|session \d{8}|as of 20\d\d|"
                            r"fixed \d|landed \d{4}|pass \d{4}|drafted 20\d\d|today|yesterday)\b", re.I)
_JUNK = re.compile(r"^(#{1,6}\s|=====|----|\s*$)|traceback|debug log", re.I)
_TOPIC_WORDS = (
    # (knowledge-file topic, keywords) used when stale entries are moved out of the always-on file.
    # Edit to match your world.
    ("service", ("service", "unit", "timer", "daemon", "restart", "port", "socket")),
    ("infra", ("server", "host", "dns", "proxy", "firewall", "cron", "systemd", "docker")),
    ("sites", ("site", "theme", "plugin", "seo", "analytics", "newsletter", "feed")),
    ("media", ("image", "video", "audio", "render", "model", "lora", "prompt", "encode")),
    ("writing", ("book", "chapter", "draft", "cover", "publish", "series", "manuscript")),
)


def topic_of(entry: str) -> str:
    low = entry.lower()
    for topic, words in _TOPIC_WORDS:
        if any(w in low for w in words):
            return topic
    return "misc"


def rule_pass(entries: list[str], keep_after_days: int = 30) -> tuple[list[str], list[tuple[str, str]]]:
    """(surviving entries, [(entry, reason)] to move out). Dated history and junk are not compressed -
    they are relocated, because no 8B can reliably rewrite 50k chars into 3k."""
    cutoff = time.time() - keep_after_days * 86400
    keep, move = [], []
    for e in entries:
        if _JUNK.search(e[:80]):
            move.append((e, "junk"))
            continue
        dates = _DATED.findall(e)
        if dates and _HISTORY_WORDS.search(e):
            newest = max((time.mktime(time.strptime(d, "%Y-%m-%d")) if len(d) == 10
                          else time.mktime(time.strptime(d + "-01", "%Y-%m-%d")) for d in dates), default=0)
            if newest < cutoff:
                move.append((e, "historical"))
                continue
        keep.append(e)
    return keep, move


def load_sleep():
    import importlib.util
    spec = importlib.util.spec_from_file_location("msleep", os.path.join(TOOLKIT, "memory_sleep.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["msleep"] = mod
    spec.loader.exec_module(mod)
    return mod


PROMPT = """Tighten the numbered memory entries below. Reply with the JSON object immediately - no analysis, no commentary, no preamble:
{"compress": [{"i": <index>, "text": "<tighter rewrite>"}]}

Rules:
- Keep every binding instruction (NEVER/ALWAYS/HARD RULE/never change/never delete) - reword it shorter, never drop it.
- No examples, no rationale, no dates older than a month.
- Each rewrite must be shorter than the original. Omit any entry you would not change.
- One entry per output item, using the index shown in brackets.
"""


def model_pass(entries: list[str], models: str, model: str, batch_chars: int = 5000,
               max_batches: int = 0) -> dict[int, str]:
    out: dict[int, str] = {}
    if not models:
        return out
    sleep = load_sleep()
    try:
        batches, current, size = [], [], 0
        for i, e in enumerate(entries):
            if size + len(e) > batch_chars and current:
                batches.append(current)
                current, size = [], 0
            current.append((i, e))
            size += len(e)
        if current:
            batches.append(current)
        if max_batches and len(batches) > max_batches:
            print("   %d batches needed; doing %d this night (token budget on a slow model)"
                  % (len(batches), max_batches))
            batches = batches[:max_batches]
        for n, batch in enumerate(batches, 1):
            ask = PROMPT + "\n\n" + "\n\n".join("[%d] %s" % (i, e) for i, e in batch)
            try:
                plan, endpoint = sleep.call_local(models, model, ask, max_tokens=3000, attempt_timeout=300,
                                                  extra_payload={"reasoning_effort": os.environ.get("HOLO_REASONING_EFFORT", "none")})
            except Exception as e:
                print("   batch %d/%d: model failed (%s: %s) - leaving those entries alone"
                      % (n, len(batches), type(e).__name__, str(e)[:140]))
                continue
            for item in plan.get("compress", []):
                try:
                    i, text = int(item["i"]), str(item["text"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= i < len(entries) and 20 <= len(text) < len(entries[i]):
                    out[i] = text
            print("   batch %d/%d via %s: %d rewrites (running total %d)" % (n, len(batches), endpoint, len(out), len(out)))
    except Exception as e:
        print("model pass aborted (%s: %s) - keeping rule-pass result" % (type(e).__name__, e))
    return out


def main() -> int:
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=os.path.join(home, "memories", "MEMORY.md"))
    ap.add_argument("--target-chars", dest="target", type=int, default=3000)
    ap.add_argument("--archive-dir", dest="archive_dir", default=os.path.join(TOOLKIT, "sleep", "memory-archive"))
    ap.add_argument("--knowledge-dir", dest="knowledge_dir", default=os.path.join(TOOLKIT, "sleep", "knowledge"))
    ap.add_argument("--model-url", default=os.environ.get("SLEEP_MODEL_URLS", ""))
    ap.add_argument("--model", default=os.environ.get("SLEEP_MODEL", "local-model"))
    ap.add_argument("--batch-chars", dest="batch_chars", type=int, default=3000,
                    help="max chars per model request (smaller keeps the JSON reply under the token cap)")
    ap.add_argument("--metrics-json", dest="metrics_json", default="",
                    help="write stage metrics for the adaptive layer")
    ap.add_argument("--no-model", dest="no_model", action="store_true", help="rule pass only")
    ap.add_argument("--max-batches", dest="max_batches", type=int, default=0,
                    help="cap model calls per file per night (0 = all) - the token budget matters on a slow box")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    raw = io.open(a.file, encoding="utf-8", errors="replace").read()
    entries = [e.strip() for e in raw.split("\u00a7") if e.strip()]
    size = len(DELIM.join(entries))
    print("%s: %d entries, %d chars (target %d)" % (os.path.basename(a.file), len(entries), size, a.target))
    if size <= a.target:
        print("already within target - nothing to do")
        return 0

    keep, move = rule_pass(entries)
    moved_chars = sum(len(e) for e, _ in move)
    print("rule pass: %d entries stay, %d move out (%d chars) by rule" % (len(keep), len(move), moved_chars))
    rewrites = {} if a.no_model else model_pass(keep, a.model_url, a.model, a.batch_chars,
                                               max_batches=a.max_batches)

    result = [rewrites.get(i, e) for i, e in enumerate(keep)]
    body = DELIM.join(result)
    print("planned: %d -> %d chars" % (size, len(body)))
    if a.dry_run:
        for e, reason in move[:8]:
            print("   would move [%s] -> knowledge/%s.md: %s" % (reason, topic_of(e), " ".join(e.split())[:75]))
        return 0
    if len(body) > max(a.target * 1.15, 800):
        print("planned result still over target - writing the rule-pass result and flagging it for the next night")

    os.makedirs(a.archive_dir, exist_ok=True)
    os.makedirs(a.knowledge_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = os.path.join(a.archive_dir, "%s-pre-rem-%s.md" % (os.path.basename(a.file), stamp))
    shutil.copy2(a.file, archive)                      # copy first: nothing leaves the disk
    by_topic: dict[str, list[tuple[str, str]]] = {}
    for e, reason in move:
        by_topic.setdefault(topic_of(e), []).append((e, reason))
    written = []
    for topic, items in by_topic.items():
        path = os.path.join(a.knowledge_dir, "%s.md" % topic)
        with io.open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n\n## Moved out of always-on memory by REM %s\n\n" % stamp)
            for e, reason in items:
                f.write("- (%s) %s\n" % (reason, " ".join(e.split())))
        written.append(path)
    with io.open(a.file, "w", encoding="utf-8", newline="\n") as f:
        f.write(body + "\n")
    print("%s now %d chars | archive %s | knowledge %s"
          % (os.path.basename(a.file), len(body), archive, ", ".join(written) or "none"))
    if a.metrics_json:
        with io.open(a.metrics_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"file": os.path.basename(a.file), "chars_before": size, "chars_after": len(body),
                       "target": a.target, "moved": len(move), "rewrites": len(rewrites)}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
