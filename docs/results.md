# Results from a real deployment

Generalised measurements from the deployment this toolkit was extracted from: one always-on host
running several Hermes homes, plus a desktop that is not always on. No configuration values, paths or
content are included — only what changed and by how much.

## The original problem, quantified

| | Before | After |
|---|---|---|
| Always-on memory injected per turn | ~49,900 chars (~12,500 tokens), 99% of its cap | ~3,200 chars (~800 tokens) for rules, ~1,800 (~445) for the profile |
| Effective cost per turn | ~19,700 tokens of memory | **~1,250 tokens** (≈94% reduction) |
| New memories | **refused** — cap reached, writes silently discarded | accepted, with headroom |
| Corpus reachable | whatever fit in that file | **8,297 facts / 1,204 entities / 12,266 links** (one home); 3,563 and 382 in two others |

The refusal is the important row. The complaint that started this work was "you can't do a process in
a new chat that we just did in another chat" — which is exactly what a full memory file produces,
because the write is rejected and the next session starts without it.

## What the deep store bought

- **Cross-corpus recall with real hits.** A keyword query returned the human's own sentence from a
  conversation months earlier alongside a curated note; an entity probe returned both notes and
  conversation lines about the same subject, ranked together.
- **Entity algebra that keyword search cannot do.** Direct component scoring separated matching from
  non-matching facts by **+0.53** (top five correct) where the provider's own probe path managed
  **+0.001** — i.e. noise. A two-entity `reason` query returned the two facts holding both, then
  dropped by an order of magnitude; on a corpus lacking that intersection it correctly returned
  nothing.
- **Storage that costs nothing per turn**: ~3-5k facts per home, queried on demand.

## Nightly REM, measured

| Stage | Result |
|---|---|
| Extraction (cloud model) | **27 facts in 90 s** for one home, 20 in 190 s for another, unattended on timers |
| Extraction (same prompt, small CPU model) | **0 facts in 41 minutes** — kept only as a fallback |
| Compression | **83 rewrites in 28 s** and **96 in 31 s**; the two always-on files came down from 51.7 KB and 51.4 KB to **46.7 KB and 48.0 KB — under the cap that had been blocking writes** |
| Rebank | all banks ≥ **2.3** SNR after packing (one category bank of 4,157 items had measured **1.0**, i.e. degraded) |
| Session history load | 897 sessions in 381 s (+867 headers, +3,079 content chunks); a further 40–60 sessions/night of deep backfill |
| Second corpus load | 828 conversations → ~3,200 facts in ~4 minutes |

## What did not help

- **gzip.** The store is dominated by dense float32 phase vectors: whole-file gzip reached 76% and
  per-blob zlib 11%, both requiring the store to be offline or decompressed per query. The text is
  only ~3 MB of ~80 MB. Re-encoding the vectors to 16-bit is the only real saving (~50%).
- **Superposed category banks as a primary mechanism.** Capacity is bounded
  (`sqrt(dim/items) >= 2`), so they needed constant re-packing, and they added little over per-fact
  vectors. Useful as a summary, not as the retrieval path.
- **Local-only inference.** The idea is most attractive offline, but the local model measured ~3
  tokens/second on that host and produced nothing usable for batch extraction. Cloud primary with a
  local fallback is what made the cycle practical.
- **An automated filter as a first-class feature.** It was necessary (it removed vague, unactionable
  output), but it also dropped anchored facts that merely contained filler words, and its early
  version recorded only a count, losing the rejected text permanently. Record the raw candidates and a
  reason per rejection; treat the filter as auditable, not as trust.

## Cost of the lessons

Nine distinct defects were found and fixed between "the design is sound" and "it runs by itself": a
user-agent block at the API edge, a required routing header, a reasoning model returning empty
content at every token budget, JSON truncation inside one batch, install-path discovery under profile
homes, an unrecoverable audit trail, an over-eager specificity gate, a stale tuned parameter, and two
passes contending for one local model. They are all in `gotchas.md` so the next implementation does
not pay for them twice.
