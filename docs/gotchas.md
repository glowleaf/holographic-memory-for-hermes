# Gotchas that cost real time

Each of these was found by measurement after a plausible-looking theory failed.

## Memory file limits

- **A full memory file silently refuses writes.** The tool returns an error and saves nothing; the
  next session simply does not know what you learned. If the agent "forgets" recent work, check the
  file size and the cap before anything else.
- The cap is **headroom, not content**. Raising it prevents refusals but does not reduce what is
  injected every turn — that is the file itself.

## The provider plugin

- `store.py` in some builds is missing `from __future__ import annotations` while annotating a class
  name in a return type: fine on Python 3.14, **NameError on 3.11**. Load it with the future import
  prepended instead of editing the installation.
- **The bundled entity extractor only matches multi-word Capitalized Phrases.** Single-word names
  are never linked, so `probe` cannot find them. Supply your own vocabulary and encode the vector
  against it; order it with curated names first, because ranking by rarity picks filenames and drops
  the names that appear everywhere.
- **The plugin's probe/reason scoring measures as noise** (separation +0.001). Score directly against
  the bound entity key (+0.53). Reproduce with `measure_probe.py` before trusting any retrieval.
- Without numpy in the executing interpreter, HRR weights are silently redistributed to keyword-only.
- A Hermes update can replace the venv, empty an untracked plugin directory and drop
  `memory.provider` from the config: after any update, re-check `hermes memory status`.

## Ingestion and retrieval

- Store **atomic facts**, not paragraphs: a sentence inside a 2,000-character note has ~0 similarity
  to any query. Group whole sentences into ~260-character facts.
- Cap entities per fact (2-3); more dilutes the signal.
- Banks are bounded by `sqrt(dim/items) >= 2`. Split oversized categories instead of increasing dim
  blindly.
- Session timestamps in the state DB are **epoch floats**; comparing them to an ISO-formatted date
  matches zero rows and the pass reports "nothing new".

## Model calls

- Some gateways/CDNs reject the default `Python-urllib` User-Agent (**403**), and some require a
  routing header (a session id) in addition to the API key. Send both.
- **Reasoning models can spend the entire budget thinking and return empty content**
  (`finish_reason=length`, `content` 0 chars, thousands of characters of reasoning). For mechanical
  tasks switch deliberation off if the API supports it; otherwise shrink the task per call.
- Replies truncate at the token cap, which looks like a parse failure inside a batch. If a batch
  fails to parse while its neighbours succeed, give it more budget or fewer items per call.
- A CPU-hosted small model can be **two orders of magnitude** too slow for batch work: measure
  tokens/second before making it the primary and keep it as a fallback.
- Never hardcode an interpreter name: `python3` does not exist on Windows. Use the running
  interpreter (`sys.executable`).

## Operating the cycle

- Two passes sharing one local model must be serialised (a lock file) and staggered in time.
- A pass that lands while the model is reloading sees transient failures: probe the endpoint's health
  and retry once rather than failing the night.
- **Never log only a count.** A pass that records "kept 20/32" and discards the rejected text makes
  those items unrecoverable; write the raw candidate list and a reason per rejection. This is the
  single most expensive lesson in this repository.
- Keep the parameters the system tunes in a file (`rem_state.json`) so a stale value cannot silently
  under-run the configuration you tested.
- Watch the host's swap: a saturated swap file plus a resident local model is what turns a nightly
  pass into a stall.
