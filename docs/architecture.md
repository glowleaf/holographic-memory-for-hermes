# Architecture

## Layers

```
turn context      ┌────────────────────────────┐
                  │ MEMORY.md / USER.md        │  small, always injected (rules + pointers)
                  └────────────────────────────┘
on demand         ┌────────────────────────────┐
                  │ fact store (SQLite)        │  facts, entity links, vectors, FTS5 index
                  └────────────────────────────┘
nightly           ┌────────────────────────────┐
                  │ REM cycle                  │  extract → consolidate → compress → rebank → adapt
                  └────────────────────────────┘
```

Nothing in the deep layer costs context until something asks for it. That is the entire trick: the
agent keeps a few thousand characters of *rules* in the prompt and a corpus behind a query.

## The fact store

Tables: `facts` (content, category, tags, trust, `hrr_vector`), `entities`, `fact_entities`,
`facts_fts` (FTS5, kept in sync by triggers), `memory_banks` (superposed category vectors).

Retrieval is hybrid and trust-weighted:

- keyword candidates from FTS5, normalised rank;
- Jaccard similarity over tokens;
- **Holographic Reduced Representations** (phase-encoded vectors, `bind` = phase addition,
  `unbind` = phase subtraction, `bundle` = circular mean) for entity-aware scoring.

Score each fact directly against the entity key:

```python
sim(fact_vector, bind(encode_atom(entity), ROLE_ENTITY))
```

Do **not** use the unbind-and-compare-to-re-encoded-text path: measured over a few hundred facts it
separated matching from non-matching by **+0.001** (noise), while direct component scoring separated
them by **+0.53** with the top five correct. `measure_probe.py` reproduces that measurement on your
own store.

`reason` composes: for several entities it takes the **minimum** of their component scores (AND
semantics), which correctly returns nothing when the corpus has no fact holding all of them.

## Quantisation and banks

Atoms are derived deterministically from a hash and quantised to 16-bit angles, so vectors are
reproducible across processes and machines. Many facts can be superposed into a *bank* per category,
but capacity is bounded by signal-to-noise:

```
SNR = sqrt(dim / items)          usable above ~2.0
```

A bank of 4,157 items at dim 4096 measured **1.0** — degraded. Keep banks under a few hundred items:
`holo.py rebank` caps them and packs consecutive days into separate buckets, while facts keep a
valid category value so nothing downstream breaks.

## The REM cycle

`rem_cycle.py`, one pass per home, stages in order; a failing stage logs and the rest continue:

1. **inbox** — ingest any `.md`/`.txt` dropped into `inbox/` (this is also where the cycle's own
   escalations land, so repeated failures become material it reads).
2. **extract** — read sessions since a watermark, ask a model for `{facts, procedures,
   contradictions}`, then filter. Facts that carry no concrete anchor (number, path, filename,
   handle, proper noun) are rejected; prose that merely *contains* filler words is kept.
3. **compress** — rule pass moves dated/historical entries and log junk to `knowledge/<topic>.md`; a
   batched model pass then tightens what remains. The full pre-REM file is archived first. A verify
   pass restores the original if the result is not under target.
4. **deep** — fully ingest N old sessions that only have a capped header (a bounded backlog).
5. **rebank** — repack banks to stay above the SNR floor.
6. **adapt** — score the run and rewrite the next run's parameters.

A lock file serialises passes, because two homes often share one local model.

## Self-improvement

Metrics are appended to `rem_metrics.jsonl`; `adapt()` maps them to parameter changes:

| Trigger (measured) | Action for the next run |
|---|---|
| extraction slow or failed | fewer chunks, smaller token budget |
| extraction fast but empty | more chunks, larger budget |
| filter rejecting nearly everything | more context per fact |
| stored facts not surviving a week | stricter specificity |
| compression missing target | more batches per file |
| a stage failing repeatedly | back that stage off, escalate into the inbox |
| deep backlog large / drained | more / fewer sessions per night |
| bank SNR below floor | smaller banks |

The parameters live in `rem_state.json`, so what the system believes about its own tuning is
inspectable, diffable and reversible.
