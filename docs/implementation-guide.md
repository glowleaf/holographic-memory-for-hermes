# Implementation guide (for a Hermes agent)

Work top to bottom. Every step has a verification command; do not proceed on an unverified step.

## 0. Preconditions

- Hermes with the bundled `holographic` memory provider available (`hermes memory status`).
- An always-on host for the nightly cycle. If your desktop is not always on, schedule REM on a box
  that is — otherwise the pass only runs on the days you happen to be sitting there.
- `numpy` **in the interpreter that actually imports the provider plugin**. Without it the vector
  maths silently degrades to keyword-only scoring.

## 1. Enable the deep store

```bash
hermes config set memory.provider holographic
hermes config set plugins.hermes-memory-store.hrr_dim 4096
hermes config set plugins.hermes-memory-store.temporal_decay_half_life 90
hermes config set plugins.hermes-memory-store.auto_extract false
hermes memory status                      # Plugin: installed, Status: available
```

Provider config is read at startup: a running session has no `fact_store` tool until it restarts.

## 2. Keep the always-on layer small

Target a few thousand characters for `MEMORY.md`, less for `USER.md`. Content that belongs there:
identity, environment pointers, hard rules, and *the instruction to query the store*. Everything else
goes in the store, a per-topic markdown file, or a skill.

Set the caps so a write is never refused, but check the file size, not just the cap — the cap is
headroom, the file is what costs context every turn.

## 3. Install the tools

```bash
cp -r tools ~/holographic_memory
export HERMES_HOME=~/.hermes
export HERMES_INSTALL=/path/to/hermes-agent       # repo containing plugins/memory/holographic
python ~/holographic_memory/holo.py init
python ~/holographic_memory/holo.py stats
```

`holo.py` discovers the Hermes install automatically (env → `$HERMES_HOME/hermes-agent` →
grandparent, for profile homes → common locations). Set `HERMES_INSTALL` explicitly in any
scheduler unit anyway.

## 4. Load what you already have

```bash
python ~/holographic_memory/holo.py ingest ~/notes/*.md
python ~/holographic_memory/ingest_sessions.py --state-db ~/.hermes/state.db
python ~/holographic_memory/ingest_sessions.py --deep 40 --deep-chars 40000   # bounded backlog
```

Ingest is resumable: already-loaded sessions are skipped by tag. Facts are stored atomically (one
sentence-group each, not paragraphs) with at most 2-3 entities — both matter for retrieval.

## 5. Verify retrieval before scheduling anything

```bash
python ~/holographic_memory/measure_probe.py ~/.hermes/memory_store.db SomeName
python ~/holographic_memory/demo.py --entity SomeName --query "a topic you remember"
```

`measure_probe.py` prints matching vs non-matching means for two scorings; the direct component
scoring must separate clearly. If it does not, stop and fix retrieval before adding data.

## 6. Schedule the REM cycle

```bash
cp systemd/holographic-rem@.service ~/.config/systemd/user/
cp systemd/holographic-rem@.timer   ~/.config/systemd/user/
# edit the placeholders: HERMES_HOME, HERMES_INSTALL, HOLO_TOOLKIT, HOLO_TAG, SLEEP_MODEL_URLS
systemctl --user daemon-reload
systemctl --user enable --now holographic-rem@default.timer
systemctl --user list-timers | grep holographic
```

For a second Hermes home (another profile), instantiate the unit with its own environment and give it
a **different** `OnCalendar` time: two passes must not share one local model.

On a host that is not always on, do not schedule a fixed hour. Run a wrapper hourly that checks a
stamp file and performs the cycle only when the last successful run is older than your interval.

## 7. Teach the agent to use it

Two edits, both required — the store does nothing by itself:

1. **Always-on memory entry** (the habit), in your own words:

   > Before answering anything about earlier work, query the deep store (`fact_store`
   > probe/search/reason, or `<toolkit>/holo.py search`). Never say "I don't remember" without
   > querying. Session search covers raw transcripts; procedures belong in skills.

2. **Skill document** describing the store, its commands and its failure modes, so a fresh session
   can act on it without re-deriving the design.

## 8. Watch it for a week

```bash
cat ~/holographic_memory/rem_metrics.jsonl      # one row per stage per run
cat ~/holographic_memory/rem_state.json         # what it currently believes its parameters should be
tail -50 ~/holographic_memory/rem.log
```

Success looks like: extraction rows with `facts_added > 0`, compression rows whose `chars_after` is
below `chars_before`, bank SNR above 2, and no escalating stage. Anything else is a bug to fix before
adding more data.
