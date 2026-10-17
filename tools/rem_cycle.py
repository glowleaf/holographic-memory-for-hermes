#!/usr/bin/env python3
"""REM cycle - nightly brain maintenance for ONE Hermes home (the always-on host runs it twice:
once for the box's own agent, once for Profile2's profile, on separate timers).

Stages, in order; a failing stage logs and the rest continue, so one broken step never blocks the night:
  1. INBOX     ingest new .md/.txt dropped in <toolkit>/inbox into this home's holographic store
  2. REM       read the day's sessions, extract durable facts with the LOCAL model, dedupe /
               reinforce / decay / prune, write a note (procedures are listed for skills)
  3. COMPRESS  shrink MEMORY.md / USER.md toward the target: tighten wording, drop useless entries,
               MOVE stale ones to knowledge files (archive + copy first, never delete)
  4. DEEP      fully ingest HOLO_DEEP old sessions that only have a capped header
  5. REBANK    repack banks so SNR stays >= 2 (superposition limit)
  6. REPORT    per-stage lines into rem.log
Everything is configurable by env so the same file serves both homes.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rem_adaptive as adaptive                      # noqa: E402  (the self-improvement layer)

TOOLKIT = os.environ.get("HOLO_TOOLKIT", os.path.expanduser("~/holographic_memory"))
HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
TAG = os.environ.get("HOLO_TAG", os.path.basename(os.path.dirname(HOME)) or "default")
PY = os.environ.get("HOLO_PYTHON", sys.executable)   # never "python3": that name doesn't exist on Windows
DB = os.environ.get("HOLO_DB", os.path.join(HOME, "memory_store.db"))
STATE = os.environ.get("HOLO_STATE_DB", os.path.join(HOME, "state.db"))
HOURS = os.environ.get("HOLO_SLEEP_HOURS", "26")
DEEP = os.environ.get("HOLO_DEEP", "60")
DEEP_CHARS = os.environ.get("HOLO_DEEP_CHARS", "40000")
TARGET = int(os.environ.get("HOLO_MEMORY_TARGET", "3000"))
MODELS = os.environ.get("SLEEP_MODEL_URLS",
                        "https://api.example.invalid/v1|provider-model|PROVIDER_API_KEY,"
                        "http://127.0.0.1:8011|local-model")   # cloud main, local 8B fallback
LOG = os.environ.get("HOLO_LOG", os.path.join(TOOLKIT, "rem.log"))
INBOX = os.environ.get("HOLO_INBOX", os.path.join(TOOLKIT, "inbox"))

env = dict(os.environ, HERMES_HOME=HOME, HOLO_TOOLKIT=TOOLKIT, SLEEP_MODEL_URLS=MODELS, PYTHONUNBUFFERED="1")
METRICS_DIR = os.path.join(TOOLKIT, "sleep", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

state = adaptive.load_state()                        # parameters come from the scoreboard, not constants
P = {k: state[k] for k in ("chunk_chars", "max_chunks", "max_tokens", "deep", "deep_chars",
                           "compress_passes", "bank_cap", "specificity", "compress_batches")}
P["deep"] = int(os.environ.get("HOLO_DEEP", P["deep"]))          # a unit may still pin these
P["deep_chars"] = int(os.environ.get("HOLO_DEEP_CHARS", P["deep_chars"]))
os.environ["HOLO_SPECIFICITY"] = str(P["specificity"])


def run(stage, label, cmd):
    """Run one stage, record its metrics for the adaptive layer, escalate after 3 straight failures."""
    metrics_path = os.path.join(METRICS_DIR, "%s-%s.json" % (TAG, stage))
    if "--metrics-json" not in cmd:
        cmd = cmd + ["--metrics-json", metrics_path]
    started = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    seconds = time.time() - started
    out = ((r.stdout or "") + (("\nSTDERR: " + r.stderr) if (r.stderr or "").strip() else "")).strip()
    metrics = {}
    try:
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, ValueError):
        pass
    for clash in ("stage", "exit_code", "seconds"):    # a metrics key must never shadow a positional arg
        metrics.pop(clash, None)

    if stage == "compress" and metrics:              # scoreboard keeps the worst file of the pass
        prev = [r_ for r_ in adaptive.history("compress") if r_.get("tag") == TAG]
        if prev and prev[-1].get("chars_after", 0) > metrics.get("chars_after", 0):
            metrics = prev[-1]
    if stage == "rebank" and not metrics:
        metrics = {"bank_cap": P["bank_cap"]}
    adaptive.record(stage, r.returncode, seconds, **metrics)

    if r.returncode != 0 and stage != "inbox":
        if adaptive.note_failure(state, stage):
            path = adaptive.escalate(INBOX, stage, out[-2000:])
            print("[%s] %-14s FAILED 3x -> escalated to %s" % (TAG, label, os.path.basename(path)), flush=True)
    else:
        adaptive.clear_failure(state, stage)

    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n===== [%s] %s | %s | exit %d | %.0fs | %s =====\n%s\n"
                % (TAG, time.strftime("%Y-%m-%d %H:%M:%S"), label, r.returncode, seconds,
                   json.dumps(metrics) if metrics else "-", out[-6000:]))
    print("[%s] %-14s exit %d  %.0fs" % (TAG, label, r.returncode, seconds), flush=True)
    return r.returncode


def backed_off(stage: str) -> bool:
    return state.get("backoff", {}).get(stage, 0) > time.time()


def _service_ok() -> bool:
    """Cheap liveness probe of the first configured model endpoint (no model call, just /health)."""
    url = MODELS.split(",")[0].split("|")[0].rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    try:
        import urllib.request
        with urllib.request.urlopen(url + "/health", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def clear_expired_backoff() -> None:
    for stage, until in list(state.get("backoff", {}).items()):
        if until <= time.time():
            state["backoff"].pop(stage, None)


codes = {}
clear_expired_backoff()

# One lock for the whole toolkit: the box pass and Profile2's pass share ONE CPU model (~3 t/s), so they
# must never run at the same time. Wait briefly for the other pass, then give up this night.
LOCK = os.path.join(TOOLKIT, "rem.lock")
try:
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) > 7200:
        os.unlink(LOCK)                                   # stale lock from a killed run
except OSError:
    pass
deadline = time.time() + int(os.environ.get("HOLO_LOCK_WAIT_S", "1200"))
while True:
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, ("%s %s\n" % (TAG, time.strftime("%Y-%m-%d %H:%M:%S"))).encode())
        os.close(fd)
        break
    except FileExistsError:
        if time.time() > deadline:
            print("[%s] another REM pass holds the lock - skipping this night" % TAG, flush=True)
            sys.exit(0)
        print("[%s] waiting for the other REM pass to finish..." % TAG, flush=True)
        time.sleep(60)

print("REM cycle for %s (home=%s, db=%s) | params: %d chunks x %d chars, max_tokens %d, deep %d, bank_cap %d, specificity %d"
      % (TAG, HOME, DB, P["max_chunks"], P["chunk_chars"], P["max_tokens"], P["deep"], P["bank_cap"], P["specificity"]), flush=True)

# 1. INBOX
os.makedirs(INBOX, exist_ok=True)
ledger = os.path.join(INBOX, ".processed-%s" % TAG)
done = set(l.strip() for l in open(ledger, encoding="utf-8")) if os.path.exists(ledger) else set()
files = [os.path.join(INBOX, f) for f in sorted(os.listdir(INBOX))
         if f.lower().endswith((".md", ".txt")) and f not in done]
if files:
    codes["inbox"] = run("inbox", [PY, os.path.join(TOOLKIT, "holo.py"), "--db", DB, "ingest"] + files)
    if codes["inbox"] == 0:
        with open(ledger, "a", encoding="utf-8") as f:
            for p in files:
                f.write(os.path.basename(p) + "\n")
else:
    print("[%s] inbox          nothing new" % TAG, flush=True)

# 2. REM extraction + consolidation (sizes come from the adaptive state)
if backed_off("rem"):
    print("[%s] rem            backed off for this night" % TAG, flush=True)
else:
    rem_cmd = [PY, os.path.join(TOOLKIT, "memory_sleep.py"), "--db", DB,
               "--state-db", STATE, "--workdir", TOOLKIT, "--hours", HOURS,
               "--chunk-chars", str(P["chunk_chars"]), "--max-tokens", str(P["max_tokens"]),
               "--max-chunks", str(P["max_chunks"]), "--attempt-timeout", "300"]
    codes["rem"] = run("rem", "rem", rem_cmd)
    # The local model is a CPU box that reloads its weights after a restart (or an OOM) for ~20s;
    # a pass that lands in that window sees 503/RemoteDisconnected. Wait it out and try once more.
    if codes["rem"] != 0 and not _service_ok():
        print("[%s] rem            model service unavailable - waiting 90s and retrying once" % TAG, flush=True)
        time.sleep(90)
        codes["rem"] = run("rem", "rem (retry)", rem_cmd)

# 3. COMPRESS the always-on files: rule pass moves dated/junk entries to knowledge files, then the
#    model tightens what is left (batched to fit a small context window). Runs compress_passes times
#    per night when the target keeps being missed.
if backed_off("compress"):
    print("[%s] compress       backed off for this night" % TAG, flush=True)
else:
    for name, target in (("MEMORY.md", TARGET), ("USER.md", TARGET * 2 // 3)):
        path = os.path.join(HOME, "memories", name)
        if not os.path.isfile(path):
            continue
        for attempt in range(int(P["compress_passes"])):
            codes["compress-" + name] = run("compress", "compress %s%s" % (name, "" if attempt == 0 else " (2)"),
                [PY, os.path.join(TOOLKIT, "memory_trim.py"), "--file", path, "--target-chars", str(target),
                 "--model-url", MODELS, "--batch-chars", os.environ.get("HOLO_BATCH_CHARS", "5000"),
                 "--max-batches", str(P["compress_batches"])])

# 4. DEEP history backlog
codes["deep"] = run("deep", "deep", [PY, os.path.join(TOOLKIT, "ingest_sessions.py"), "--db", DB,
                                     "--state-db", STATE, "--deep", str(P["deep"]),
                                     "--deep-chars", str(P["deep_chars"])])

# 5. REBANK
codes["rebank"] = run("rebank", "rebank", [PY, os.path.join(TOOLKIT, "holo.py"), "--db", DB, "rebank",
                                           "--max-per-bank", str(P["bank_cap"])])

# 6. ADAPT - score this run and rewrite tomorrow's parameters
state, reasons = adaptive.adapt(state, DB)
adaptive.save_state(state)
if reasons:
    print("[%s] ADAPTED for the next run:" % TAG, flush=True)
    for why in reasons:
        print("[%s]   - %s" % (TAG, why), flush=True)
else:
    print("[%s] adapted: no change warranted" % TAG, flush=True)

print("[%s] summary: %s" % (TAG, ", ".join("%s=%d" % kv for kv in codes.items())), flush=True)
try:
    os.unlink(LOCK)
except OSError:
    pass
fatal = [k for k in ("deep", "rebank") if codes.get(k, 0) != 0]
sys.exit(1 if fatal else 0)
