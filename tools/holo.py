#!/usr/bin/env python
"""holo.py - holographic memory toolkit (Hermes 'holographic' provider, loaded standalone).

Loads the plugin's MemoryStore without booting the agent, so it works from cron on any box.

  init        create/verify the SQLite store
  add         store one fact               (--content, --category, --tags)
  ingest      store many facts from MD files (split on '---' or the section sign)
  search      hybrid FTS5 + Jaccard + HRR keyword search
  probe       all facts about an entity (algebraic unbind)
  reason      facts where ALL given entities play a role (vector JOIN)
  contradict  conflicting-claim pairs (memory hygiene)
  stats       counts, per-category banks, HRR SNR headroom
  sleep       consolidation pass: dedupe/reinforce candidates, decay, prune to MD, rebuild banks, log

Source of the store lives in the Hermes install: plugins/memory/holographic/."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import sqlite3
import sys
import types
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_install() -> str:
    """Locate the Hermes install that contains the holographic plugin.

    A hardcoded Windows default meant any run on the always-on host that did not export HERMES_INSTALL
    failed with 'plugin not found at <HOME>/...' — the scheduled units set it, ad-hoc runs
    did not. Probe the likely roots and use the first that really holds the plugin."""
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    candidates = [os.environ.get("HERMES_INSTALL", ""), os.environ.get("HOLO_INSTALL", ""),
                  os.path.join(home, "hermes-agent"),
                  os.path.join(os.path.dirname(os.path.abspath(home.rstrip("/\\"))), "hermes-agent"),
                  # a PROFILE home is <hermes>/.hermes/profiles/<name>, so the install is two levels up
                  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(home.rstrip("/\\")))), "hermes-agent"),
                  os.path.expanduser("~/hermes-agent"),
                  r"<HERMES_INSTALL>",
                  os.path.dirname(HERE)]
    for cand in candidates:
        if cand and os.path.isdir(os.path.join(cand, "plugins", "memory", "holographic")):
            return cand
    return next((c for c in candidates if c), ".")


DEFAULT_INSTALL = _find_install()
DEFAULT_DB = os.environ.get("HERMES_HOME", r"<HERMES_HOME>") + "/memory_store.db"
CATEGORIES = ("user_pref", "project", "tool", "general")


def load_plugin(install: str):
    """Import plugins/memory/holographic/{holographic,store,retrieval}.py as a synthetic package.

    store.py is missing ``from __future__ import annotations`` while using a class-name return
    annotation (``def __enter__(self) -> MemoryStore``), which NameErrors on Python < 3.14 where
    annotations are still evaluated eagerly. Prepend the future import instead of editing the install."""
    root = os.path.join(install, "plugins", "memory", "holographic")
    if not os.path.isdir(root):
        sys.exit("holographic plugin not found at %s" % root)
    pkg = types.ModuleType("holo_pkg")
    pkg.__path__ = [root]
    sys.modules["holo_pkg"] = pkg
    for name in ("holographic", "store", "retrieval"):
        path = os.path.join(root, name + ".py")
        src = io.open(path, encoding="utf-8").read()
        if "from __future__ import annotations" not in src:
            src = "from __future__ import annotations\n" + src
        spec = importlib.util.spec_from_loader("holo_pkg." + name, loader=None)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["holo_pkg." + name] = mod
        exec(compile(src, path, "exec"), mod.__dict__)
    return sys.modules["holo_pkg.store"], sys.modules["holo_pkg.retrieval"], sys.modules["holo_pkg.holographic"]


STORE, RETRIEVAL, HRR = load_plugin(DEFAULT_INSTALL)
if not HRR._HAS_NUMPY:
    print("WARNING: numpy missing - HRR algebra degrades to keyword-only scoring", file=sys.stderr)


def vec_score(fact_vec, term: str, dim: int, role: str | None = "entity") -> float:
    """Similarity of a fact vector to a term's bound component (the ONLY scoring that works).

    Measured on 254 facts (query 'Profile2'): the bundled retriever's probe path
    (sim(unbind(fact,i>bind(e,ROLE_ENTITY)), bind(encode_text(content),ROLE_CONTENT))) scores
    matching facts -0.004 vs non-matching -0.006 -> separation +0.001, i.e. noise. Scoring the
    fact against bind(e, ROLE_ENTITY) directly gives matching 0.530 vs non-matching -0.000
    -> separation +0.530 with 100% precision in the top 5."""
    if role == "entity":
        key = HRR.bind(HRR.encode_atom(term.lower(), dim), HRR.encode_atom(HRR.ROLE_ENTITY, dim))
    elif role == "content":
        key = HRR.bind(HRR.encode_atom(term.lower(), dim), HRR.encode_atom(HRR.ROLE_CONTENT, dim))
    else:                                                        # bare atom: structural adjacency
        key = HRR.encode_atom(term.lower(), dim)
    return HRR.similarity(fact_vec, key)


def rank_by_vectors(st, terms: list[str], mode: str, category: str | None, limit: int):
    """Vector-space ranking over the store: probe (one entity), reason (ALL entities), related (bare).
    Falls back to keyword search when no vectors exist (no numpy at write time)."""
    where = "WHERE hrr_vector IS NOT NULL" + (" AND category = ?" if category else "")
    rows = st._conn.execute("SELECT fact_id, content, category, tags, trust_score, hrr_vector FROM facts " + where,
                            [category] if category else []).fetchall()
    if not rows:
        return []
    scored = []
    for r in rows:
        vec = HRR.bytes_to_phases(r["hrr_vector"], dim=st.hrr_dim)
        if mode == "probe":
            sim = vec_score(vec, terms[0], st.hrr_dim, "entity")
        elif mode == "related":
            sim = max(vec_score(vec, terms[0], st.hrr_dim, "entity"), vec_score(vec, terms[0], st.hrr_dim, None))
        else:                                                    # reason: AND over entities (min)
            sim = min(vec_score(vec, t, st.hrr_dim, "entity") for t in terms)
        scored.append((sim, dict(r)))
    scored.sort(key=lambda x: -x[0])
    return [dict(f, score=round(s, 4)) for s, f in scored[:limit]]


def print_ranked(rows):
    if not rows:
        print("no vectors in the store yet (ingest first)")
        return
    for r in rows:
        print("%.3f [%.2f] #%s %s" % (r["score"], r.get("trust_score", 0), r["fact_id"],
                                      " ".join(r["content"].split())[:150]))


def store_for(args):
    return STORE.MemoryStore(db_path=args.db, default_trust=float(args.trust), hrr_dim=int(args.dim))


def retriever_for(st, args):
    return RETRIEVAL.FactRetriever(store=st, hrr_dim=int(args.dim), hrr_weight=0.3,
                                   temporal_decay_half_life=int(args.decay))


def split_facts(text: str) -> list[str]:
    """Facts are separated by a '---' rule or the '§' section sign (memory-store format).
    Leading markdown header lines belong to the file, not to the first fact, so strip them
    instead of discarding the block (dropping the header skipped the file's first entry)."""
    out = []
    for block in text.replace("\r\n", "\n").split("\n---\n"):
        for chunk in block.split("\n\u00a7\n"):
            lines = chunk.split("\n")
            while lines and (not lines[0].strip() or lines[0].startswith("#")):
                lines.pop(0)
            chunk = "\n".join(lines).strip()
            if len(chunk) >= 12:
                out.append(chunk)
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


def atomize(text: str, max_len: int = 260) -> list[str]:
    """Split a long note into atomic facts of <= max_len chars, grouping whole sentences.

    HRR retrieval is bag-of-words: burying a sentence inside a 2,000-char paragraph makes its
    similarity to any query ~0, which is what turned probe/reason into noise. Atomic facts are
    the granularity the vector algebra is designed for. Runt sentences merge into the neighbour."""
    atoms, current = [], ""
    for sentence in _SENT_SPLIT.split(" ".join(text.split())):
        sentence = sentence.strip()
        if not sentence:
            continue
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_len:
            current += " " + sentence
        else:
            atoms.append(current)
            current = sentence
    if current:
        atoms.append(current)
    merged = []
    for atom in atoms:
        if merged and len(atom) < 40:
            merged[-1] = (merged[-1] + " " + atom)[:max_len]
        else:
            merged.append(atom)
    return [a for a in merged if len(a) >= 12]


# Entity vocabulary. The bundled extractor only matches multi-word Capitalized Phrases, so
# single-word names that matter here (Profile2, accelerator-host, Ubuntu, Maestro, Nostr…) are never bound into
# the HRR vectors and probe()/reason() cannot find them. This vocabulary is applied by keyword.
VOCAB_EXTRAS = (
    # Entities this store should care about: people, hosts, services, products. Edit for your world -
    # the bundled extractor only links multi-word Capitalized Phrases, so single-word names must be
    # listed here or they can never be probed.
    "Example", "Server", "Postgres", "Redis", "Docker", "Hermes",
)
_VOCAB_STOP = frozenset(("The", "This", "That", "These", "Those", "When", "Never", "Always", "Use", "Add",
                         "Note", "Hard", "Rule", "Check", "Never", "All", "Every", "For", "And", "But",
                         "Not", "Only", "Keep", "Run", "Set", "Put", "Get", "Make", "Do", "If", "It",
                         "He", "She", "They", "We", "You", "I", "A", "An", "In", "On", "At", "To", "Of"))


def build_vocab(contents: list[str], min_hits: int = 3) -> tuple[list[str], dict[str, int]]:
    counts: dict[str, int] = {}
    for text in contents:
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_.+-]{2,}\b", text):
            if token in _VOCAB_STOP:
                continue
            counts[token] = counts.get(token, 0) + 1
    auto = {t for t, n in counts.items() if n >= min_hits}
    return sorted(auto | set(VOCAB_EXTRAS), key=lambda s: (-len(s), s)), counts


def entities_in(text: str, vocab: list[str], counts: dict[str, int] | None = None, cap: int = 3) -> list[str]:
    """Curated names present in the text first, then rare-but-real auto terms, capped.

    Ordering by raw rarity was wrong: it picked junk tokens (filenames like
    skills_prompt_snapshot.json, count 1) and DROPPED the salient names, because a name that
    appears in many facts has a high count (measured: probe stayed flat at 0.252 for every fact).
    Curated entities (VOCAB_EXTRAS) win; filename-shaped tokens are never entities."""
    counts = counts or {}
    low = text.lower()
    curated = [t for t in VOCAB_EXTRAS if t.lower() in low]
    def usable(t: str) -> bool:
        return not any(c in t for c in "._/\\") and not any(ch.isdigit() for ch in t)
    auto = [t for t in vocab if t not in curated and usable(t) and t.lower() in low]
    auto.sort(key=lambda t: (counts.get(t, 0), -len(t)))
    out: list[str] = []
    for term in curated + auto:
        if term not in out:
            out.append(term)
        if len(out) >= cap:
            break
    return out


def insert_fact_with_entities(st, content: str, category: str, tags: str, entities: list[str]):
    """Insert a fact whose HRR vector is encoded against OUR entity list, then link the entities.

    store.add_fact() encodes the vector from its own regex-extracted entities, so a fact stored
    that way can never be found by probe() for a single-word entity. Writing the row here keeps
    the install untouched while making the algebra actually work."""
    vec = HRR.encode_fact(content, entities, st.hrr_dim)
    cur = st._conn.execute(
        "INSERT INTO facts (content, category, tags, trust_score, hrr_vector) VALUES (?, ?, ?, ?, ?)",
        (content, category, tags, st.default_trust, HRR.phases_to_bytes(vec)))
    fid = cur.lastrowid
    for name in entities:
        row = st._conn.execute("SELECT entity_id FROM entities WHERE name = ?", (name,)).fetchone()
        eid = row["entity_id"] if row else st._conn.execute(
            "INSERT INTO entities (name, entity_type) VALUES (?, 'unknown')", (name,)).lastrowid
        st._conn.execute("INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)", (fid, eid))
    st._conn.commit()
    return fid


def cmd_init(a):
    st = store_for(a)
    n = st._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    st.close()
    print("store ready: %s (%d facts)" % (a.db, n))


def cmd_add(a):
    st = store_for(a)
    vocab, counts = build_vocab([r["content"] for r in st._conn.execute("SELECT content FROM facts")])
    ents = entities_in(a.content, vocab, counts, a.entities_cap)
    try:
        fid = insert_fact_with_entities(st, a.content, a.category, a.tags, ents)
        st._rebuild_bank(a.category)
        print("added fact_id=%s entities=%s" % (fid, ents))
    except sqlite3.IntegrityError:
        print("duplicate - already stored")
    finally:
        st.close()


def cmd_ingest(a):
    st = store_for(a)
    added = dup = 0
    per_file = []
    entries: list[tuple[str, str, str]] = []      # (atom, category, tags)
    for path in a.files:
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        if name == "INDEX.md":
            continue
        text = io.open(path, encoding="utf-8", errors="replace").read()
        cat = a.category
        if a.category == "auto":
            low = name.lower()
            cat = ("user_pref" if "hard-rules" in low or "user" in low
                   else "tool" if any(k in low for k in ("tool", "image", "media", "hardware", "service"))
                   else "project" if any(k in low for k in ("infra", "wordpress", "bots", "writing", "games", "shop"))
                   else "general")
        n = 0
        for i, entry in enumerate(split_facts(text)):
            for atom in (atomize(entry) if a.atomic else [entry[:2000]]):
                tags = "%s;src=%s#%d" % (a.tags, name.replace(".md", ""), i)
                entries.append((atom, cat, tags))
                n += 1
        per_file.append((name, cat, n))
    vocab, counts = build_vocab([e[0] for e in entries])
    for atom, cat, tags in entries:
        ents = entities_in(atom, vocab, counts, a.entities_cap)
        try:
            insert_fact_with_entities(st, atom, cat, tags, ents)
            added += 1
        except sqlite3.IntegrityError:
            dup += 1
    for cat in {e[1] for e in entries}:
        st._rebuild_bank(cat)
    total = st._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    st.close()
    for name, cat, n in per_file:
        print("  %-36s %-10s +%d" % (name, cat, n))
    print("ingested %d new, %d duplicates skipped; store now %d facts" % (added, dup, total))


def cmd_search(a):
    st = store_for(a)
    for r in retriever_for(st, a).search(a.query, category=a.category, min_trust=a.min_trust, limit=a.limit):
        print("%.3f [%.2f] #%s %s" % (r["score"], r["trust_score"], r["fact_id"], r["content"][:150].replace("\n", " ")))
    st.close()


def cmd_probe(a):
    st = store_for(a)
    print_ranked(rank_by_vectors(st, [a.entity], "probe", a.category, a.limit))
    st.close()


def cmd_related(a):
    st = store_for(a)
    print_ranked(rank_by_vectors(st, [a.entity], "related", a.category, a.limit))
    st.close()


def cmd_reason(a):
    st = store_for(a)
    print_ranked(rank_by_vectors(st, a.entities, "reason", a.category, a.limit))
    st.close()


def cmd_contradict(a):
    st = store_for(a)
    pairs = retriever_for(st, a).contradict(category=a.category, threshold=a.threshold, limit=a.limit)
    for p in pairs:
        print("contradiction %.2f (entity overlap %.2f) shared=%s" % (p["contradiction_score"], p["entity_overlap"], ",".join(p["shared_entities"])))
        print("  A #%s %s" % (p["fact_a"]["fact_id"], p["fact_a"]["content"][:130].replace("\n", " ")))
        print("  B #%s %s" % (p["fact_b"]["fact_id"], p["fact_b"]["content"][:130].replace("\n", " ")))
    print("%d conflicting pair(s)" % len(pairs))
    st.close()


def cmd_rebank(a):
    """Rebuild category banks with a per-bank size cap; oversized categories are partitioned by month.

    One bank per category collapses as the store grows: 4,157 chat facts in `cat:general` measured
    SNR 1.0 (limit sqrt(dim/items) >= 2.0). Superposition is meant for a few hundred items, so a big
    category is split into month banks (`chat:YYYY-MM`, read from each fact's date= tag) while the
    facts keep their valid category enum. Bank names are free-form rows, so nothing breaks the
    agent's own fact_store category filter."""
    st = store_for(a)
    c = st._conn
    created = []
    for row in c.execute("SELECT category, COUNT(*) n FROM facts GROUP BY category"):
        cat, n = row["category"] or "general", row["n"]
        if n <= a.max_per_bank:
            st._rebuild_bank(cat)
            created.append(("cat:" + cat, n))
            continue
        # too big for one bundle: remove the monolithic bank and bundle in <= cap-sized buckets.
        # Grouping by month is not enough - a heavy month (2,449 facts in 2026-09) still measured
        # SNR 1.3, so facts are grouped by day and consecutive days are packed into buckets.
        c.execute("DELETE FROM memory_banks WHERE bank_name = ?", ("cat:%s" % cat,))
        by_day: dict[str, list] = {}
        for f in c.execute("SELECT hrr_vector, tags FROM facts WHERE category = ? AND hrr_vector IS NOT NULL", (cat,)):
            m = re.search(r"date=(\d{4}-\d{2}-\d{2})", f["tags"] or "")
            by_day.setdefault(m.group(1) if m else "undated", []).append(f["hrr_vector"])
        bucket, buckets = [], []
        for day in sorted(by_day):
            if bucket and len(bucket) + len(by_day[day]) > a.max_per_bank:
                buckets.append(bucket)
                bucket = []
            bucket += [(day, v) for v in by_day[day]]
        if bucket:
            buckets.append(bucket)
        prefix = "chat" if cat == "general" else cat
        for n, chunk in enumerate(buckets, 1):
            bank = "%s:%s" % (prefix, chunk[0][0] if len(buckets) == 1 else "%s#%d" % (chunk[0][0][:7], n))
            array = HRR.bundle(*[HRR.bytes_to_phases(v, dim=st.hrr_dim) for _, v in chunk])
            c.execute("INSERT INTO memory_banks (bank_name, vector, dim, fact_count, updated_at) "
                      "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) "
                      "ON CONFLICT(bank_name) DO UPDATE SET vector=excluded.vector, dim=excluded.dim, "
                      "fact_count=excluded.fact_count, updated_at=excluded.updated_at",
                      (bank, HRR.phases_to_bytes(array), st.hrr_dim, len(chunk)))
            created.append((bank, len(chunk)))
        keep = {b for b, _ in created}
        for stale in [r["bank_name"] for r in c.execute("SELECT bank_name FROM memory_banks WHERE bank_name LIKE ?", (prefix + ":%",))]:
            if stale not in keep:
                c.execute("DELETE FROM memory_banks WHERE bank_name = ?", (stale,))
    c.commit()
    print("banks rebuilt (cap %d items/bank):" % a.max_per_bank)
    snrs = []
    for row in c.execute("SELECT bank_name, fact_count, dim FROM memory_banks ORDER BY fact_count DESC"):
        n = int(row["fact_count"] or 0)
        snr = HRR.snr_estimate(int(row["dim"]), n) if n else float("inf")
        if snr != float("inf"):
            snrs.append(snr)
        print("  %-22s %5d items  SNR %s  %s" % (row["bank_name"], n,
              ("%.1f" % snr) if snr != float("inf") else "inf", "ok" if snr >= 2 else "DEGRADED"))
    if a.metrics_json:
        import json as _json
        with io.open(a.metrics_json, "w", encoding="utf-8", newline="\n") as f:
            _json.dump({"min_snr": round(min(snrs), 2) if snrs else None, "banks": len(snrs),
                        "max_per_bank": a.max_per_bank}, f, indent=2)
    st.close()


def cmd_stats(a):
    st = store_for(a)
    c = st._conn
    total = c.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    ents = c.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    links = c.execute("SELECT COUNT(*) FROM fact_entities").fetchone()[0]
    print("facts=%d entities=%d links=%d dim=%d db=%s" % (total, ents, links, a.dim, a.db))
    for row in c.execute("SELECT category, COUNT(*), AVG(trust_score) FROM facts GROUP BY category ORDER BY 2 DESC"):
        print("  %-10s %4d facts  avg trust %.2f" % (row[0], row[1], row[2] or 0))
    banks = c.execute("SELECT bank_name, fact_count, dim FROM memory_banks ORDER BY fact_count DESC").fetchall()
    for b in banks:
        if not b["fact_count"]:
            print("  bank %-20s %5d items" % (b["bank_name"], 0))
            continue
        snr = HRR.snr_estimate(int(b["dim"]), int(b["fact_count"]))
        label = "degraded, raise --dim" if snr < 2.0 else "ok"
        print("  bank %-20s %5d items  SNR %s  %s" % (b["bank_name"], b["fact_count"],
                                                      ("%.1f" % snr) if snr != float("inf") else "inf", label))
    st.close()


def cmd_sleep(a):
    """Consolidation: candidates in -> dedupe/reinforce or add, decay stale trust, archive+prune dead facts, rebuild banks, log."""
    st = store_for(a)
    ret = retriever_for(st, a)
    c = st._conn
    report = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "added": 0, "reinforced": 0,
              "contradictions": [], "pruned": [], "banks": [], "candidates": 0}

    candidates = []
    for path in (a.candidates or []):
        if not os.path.isfile(path):
            continue
        raw = io.open(path, encoding="utf-8", errors="replace").read().strip()
        try:
            payload = json.loads(raw)
            rows = payload if isinstance(payload, list) else payload.get("facts", [])
            candidates += [r if isinstance(r, str) else r.get("content", "") for r in rows]
        except ValueError:
            candidates += split_facts(raw)
    candidates = [x.strip() for x in candidates if len(x.strip()) >= 12]
    report["candidates"] = len(candidates)

    # 1) dedupe by exact content, then by vector similarity; reinforce instead of duplicating
    for fact in candidates:
        existing = None
        try:
            existing = c.execute("SELECT fact_id, trust_score FROM facts WHERE content = ?", (fact[:2000],)).fetchone()
        except sqlite3.Error:
            pass
        if not existing:
            hits = ret.search(fact, min_trust=0.0, limit=1)
            if hits and hits[0]["score"] >= a.similarity:
                existing = {"fact_id": hits[0]["fact_id"], "trust_score": hits[0]["trust_score"]}
        if existing:
            st.update_fact(int(existing["fact_id"]), trust_delta=0.02)
            report["reinforced"] += 1
        else:
            try:
                st.add_fact(fact[:2000], category=a.category, tags=a.tags)
                report["added"] += 1
            except sqlite3.IntegrityError:
                report["reinforced"] += 1

    # 2) memory hygiene: conflicting claims
    report["contradictions"] = [{"score": p["contradiction_score"], "shared": p["shared_entities"],
                                 "a": p["fact_a"]["fact_id"], "b": p["fact_b"]["fact_id"]}
                                for p in ret.contradict(threshold=a.threshold, limit=25)]

    # 3) pruning: low-trust AND old -> archived to MD first, then removed (never a silent delete)
    if a.prune_below is not None:
        rows = c.execute("SELECT fact_id, content, category, trust_score, updated_at FROM facts "
                         "WHERE trust_score < ? ORDER BY trust_score LIMIT ?", (a.prune_below, a.prune_limit)).fetchall()
        for r in rows:
            age_days = 0
            try:
                ts = datetime.fromisoformat(r["updated_at"])
                age_days = (datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)).days
            except (ValueError, TypeError):
                pass
            if age_days >= a.prune_days:
                report["pruned"].append({"fact_id": r["fact_id"], "trust": r["trust_score"],
                                         "age_days": age_days, "content": r["content"]})
                st.remove_fact(int(r["fact_id"]))
        if report["pruned"] and a.archive_dir:
            os.makedirs(a.archive_dir, exist_ok=True)
            path = os.path.join(a.archive_dir, "pruned-%s.md" % datetime.now().strftime("%Y%m%d-%H%M%S"))
            with io.open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("# Pruned facts (archived before removal)\n\n")
                for p in report["pruned"]:
                    f.write("- trust %.2f, age %dd, was #%s: %s\n" % (p["trust"], p["age_days"], p["fact_id"], p["content"]))
            report["prune_archive"] = path

    # 4) quantize: full bank rebuild + SNR headroom per bank
    if hasattr(st, "_rebuild_bank"):
        for row in c.execute("SELECT DISTINCT category FROM facts").fetchall():
            if row[0]:
                st._rebuild_bank(row[0])
    for b in c.execute("SELECT bank_name, fact_count, dim FROM memory_banks"):
        snr = HRR.snr_estimate(int(b["dim"]), int(b["fact_count"])) if b["fact_count"] else float("inf")
        report["banks"].append({"bank": b["bank_name"], "items": b["fact_count"], "snr": round(snr, 2) if snr != float("inf") else None})
    report["total_facts"] = c.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    st.close()

    report["ended"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if a.log_dir:
        os.makedirs(a.log_dir, exist_ok=True)
        path = os.path.join(a.log_dir, "sleep-%s.md" % datetime.now().strftime("%Y-%m-%d"))
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Memory sleep pass - %s\n\n" % report["started"])
            f.write("- candidates: %d, added: %d, reinforced: %d\n" % (report["candidates"], report["added"], report["reinforced"]))
            f.write("- store total: %d facts\n" % report["total_facts"])
            f.write("- contradictions: %d\n" % len(report["contradictions"]))
            f.write("- pruned: %d%s\n" % (len(report["pruned"]), (" -> " + report.get("prune_archive", "")) if report.get("prune_archive") else ""))
            for b in report["banks"]:
                f.write("- bank %s: %s items, SNR %s\n" % (b["bank"], b["items"], b["snr"]))
        report["log"] = path
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description="holographic memory toolkit")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--dim", default=4096, type=int)
    p.add_argument("--trust", default=0.5, type=float)
    p.add_argument("--decay", default=90, type=int, help="temporal decay half-life in days (0 = off)")
    p.add_argument("--entities-cap", dest="entities_cap", default=3, type=int,
                   help="max entities bound per fact (higher dilutes probe/reason signal)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init").set_defaults(func=cmd_init)
    s = sub.add_parser("rebank")
    s.add_argument("--max-per-bank", dest="max_per_bank", type=int, default=800,
                   help="categories larger than this get split into per-month banks")
    s.add_argument("--metrics-json", dest="metrics_json", default="", help="write stage metrics for the adaptive layer")
    s.set_defaults(func=cmd_rebank)
    sub.add_parser("stats").set_defaults(func=cmd_stats)

    s = sub.add_parser("add"); s.add_argument("content"); s.add_argument("--category", default="general", choices=CATEGORIES)
    s.add_argument("--tags", default=""); s.set_defaults(func=cmd_add)

    s = sub.add_parser("ingest"); s.add_argument("files", nargs="+")
    s.add_argument("--category", default="auto"); s.add_argument("--tags", default="ingested")
    s.add_argument("--no-atomic", dest="atomic", action="store_false", default=True,
                   help="store whole paragraphs instead of atomic facts (worse vector recall)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("--category"); s.add_argument("--min-trust", dest="min_trust", type=float, default=0.0)
    s.add_argument("--limit", type=int, default=8); s.set_defaults(func=cmd_search)

    s = sub.add_parser("probe"); s.add_argument("entity"); s.add_argument("--category"); s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("related"); s.add_argument("entity"); s.add_argument("--category"); s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_related)

    s = sub.add_parser("reason"); s.add_argument("entities", nargs="+"); s.add_argument("--category"); s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_reason)

    s = sub.add_parser("contradict"); s.add_argument("--category"); s.add_argument("--threshold", type=float, default=0.3)
    s.add_argument("--limit", type=int, default=10); s.set_defaults(func=cmd_contradict)

    s = sub.add_parser("sleep"); s.add_argument("--candidates", nargs="*", default=[])
    s.add_argument("--category", default="general", choices=CATEGORIES); s.add_argument("--tags", default="sleep")
    s.add_argument("--similarity", type=float, default=0.85, help="vector similarity above which a candidate reinforces an existing fact")
    s.add_argument("--threshold", type=float, default=0.3, help="contradiction threshold")
    s.add_argument("--prune-below", dest="prune_below", type=float, default=None)
    s.add_argument("--prune-days", dest="prune_days", type=int, default=120)
    s.add_argument("--prune-limit", dest="prune_limit", type=int, default=50)
    s.add_argument("--archive-dir", dest="archive_dir", default=os.path.join(HERE, "pruned"))
    s.add_argument("--log-dir", dest="log_dir", default=os.path.join(HERE, "sleep-logs"))
    s.set_defaults(func=cmd_sleep)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
