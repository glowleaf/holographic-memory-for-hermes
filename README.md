# Holographic memory for Hermes

A small always-on memory, a large on-demand fact store, and a nightly consolidation cycle ("REM")
that keeps both healthy. This repository is the **architecture plus working standalone tools** for a
Hermes agent to implement it. It is provider-agnostic and contains no private data.

## The problem it solves

Hermes' built-in memory is a bounded file pair (`MEMORY.md` / `USER.md`) injected into every turn.
It has a character cap and **no auto-compaction**: when a write would exceed the cap the tool returns
an error and *nothing is saved*. A profile that fills up therefore stops learning — a process
established in one session silently never reaches the next one, which reads as "the agent forgot what
we did an hour ago".

The fix is not a bigger file (that costs context on every turn). It is a layered memory:

1. **Always-on layer** — a *small* rules file (target a few thousand characters), so it fits a local
   model's window and never crowds the prompt.
2. **Deep store** — a local SQLite fact store queried on demand: chat history, notes, exports.
   Costs zero context until asked.
3. **REM cycle** — a nightly pass that extracts new durable facts from the day's sessions, dedupes
   and prunes the store, compresses the always-on files, and tunes its own budgets from measured
   outcomes.

## Quick start

```bash
# 1. deep store (bundled provider, local SQLite)
hermes config set memory.provider holographic
hermes config set plugins.hermes-memory-store.hrr_dim 4096
hermes config set plugins.hermes-memory-store.temporal_decay_half_life 90
hermes memory status            # expect: holographic (local) <- active

# 2. tools
cp -r tools ~/holographic_memory
python ~/holographic_memory/holo.py init
python ~/holographic_memory/holo.py ingest ~/notes/*.md
python ~/holographic_memory/holo.py probe SomeName

# 3. nightly REM on your always-on host
cp systemd/holographic-rem@.{service,timer} ~/.config/systemd/user/
systemctl --user enable --now holographic-rem@default.timer
```

Then follow `docs/implementation-guide.md` for the full sequence, including the rules that make the
agent actually *consult* the store, and read `docs/gotchas.md` before debugging anything — it lists
the traps that cost real time here.

## Layout

| Path | What |
|---|---|
| `tools/holo.py` | store management: init, ingest, search, probe, reason, contradict, stats, rebank, sleep |
| `tools/memory_sleep.py` | REM extraction: sessions → durable facts (with a specificity gate) |
| `tools/memory_trim.py` | compress the always-on files (rule pass + batched model pass) |
| `tools/rem_cycle.py` | the orchestrator: inbox → extract → compress → deep → rebank → adapt |
| `tools/rem_adaptive.py` | self-improvement: metrics in, next run's parameters out |
| `tools/ingest_sessions.py` | load Hermes session history (tiered, resumable, deep-history mode) |
| `tools/ingest_chatgpt.py`, `tools/ingest_cursor.py` | load other conversation corpora |
| `tools/measure_probe.py` | prove a scoring metric separates matching from non-matching facts |
| `tools/demo.py` | end-to-end demonstration against a live store |
| `docs/gotchas.md` | the traps, with the measurement behind each |
| `docs/results.md` | what the deployment measured before vs after (generalised) |
| `systemd/` | unit + timer templates (one instance per Hermes home) |

## Honest limitations

- The store's vectors are dense float32 phases and dominate its size; gzip does not help (measured
  11%), int16 repacking gives ~50%. Storage is cheap; treat that as housekeeping.
- Extraction quality depends on the model. A small CPU-hosted model produced **zero** usable facts in
  41 minutes; the same pass through a cloud model produced 27 in 90 seconds. Keep the local model as
  a fallback, not the primary.
- The self-improvement layer only knows what you measure. It was blind until per-stage metrics and
  raw candidate dumps existed — see `docs/gotchas.md`, last section.
