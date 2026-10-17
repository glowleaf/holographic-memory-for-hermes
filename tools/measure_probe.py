#!/usr/bin/env python
"""Measure whether the holographic probe math can separate a matching fact from a non-matching one.

Compares two scorings for the same query entity:
  A) plugin path : sim(unbind(fact_vec, bind(e,ROLE_ENTITY)), bind(encode_text(content), ROLE_CONTENT))
  B) direct      : sim(fact_vec, bind(e, ROLE_ENTITY))   # is the entity's bound term a component of the fact?

Prints, for each scoring, the mean score of facts whose text really contains the entity vs the rest,
and the separation. Facts are ground-truthed by substring, not by the store's own entity links."""
import os
import sys
import types
import io
import importlib.util

INSTALL = os.environ.get("HERMES_INSTALL", r"<HERMES_INSTALL>")
root = os.path.join(INSTALL, "plugins", "memory", "holographic")
pkg = types.ModuleType("holo_pkg"); pkg.__path__ = [root]; sys.modules["holo_pkg"] = pkg
for name in ("holographic", "store", "retrieval"):
    src = io.open(os.path.join(root, name + ".py"), encoding="utf-8").read()
    if "from __future__ import annotations" not in src:
        src = "from __future__ import annotations\n" + src
    spec = importlib.util.spec_from_loader("holo_pkg." + name, loader=None)
    mod = importlib.util.module_from_spec(spec); sys.modules["holo_pkg." + name] = mod
    exec(compile(src, os.path.join(root, name + ".py"), "exec"), mod.__dict__)

HRR = sys.modules["holo_pkg.holographic"]
STORE = sys.modules["holo_pkg.store"]

db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "memory_store.db")
entity = sys.argv[2] if len(sys.argv) > 2 else "Profile2"
dim = 4096

st = STORE.MemoryStore(db_path=db, hrr_dim=dim)
rows = st._conn.execute("SELECT fact_id, content, hrr_vector FROM facts WHERE hrr_vector IS NOT NULL").fetchall()
print("facts with vectors: %d  (dim %d)  query entity: %r" % (len(rows), dim, entity))

key = HRR.bind(HRR.encode_atom(entity.lower(), dim), HRR.encode_atom(HRR.ROLE_ENTITY, dim))
rc = HRR.encode_atom(HRR.ROLE_CONTENT, dim)

# sanity: the algebra itself
a, b = HRR.encode_atom("alpha", dim), HRR.encode_atom("beta", dim)
print("sanity unbind(bind(a,b),a) vs b: %.3f   (should be ~1.0)" % HRR.similarity(HRR.unbind(HRR.bind(a, b), a), b))

hits, misses = [], []
for r in rows:
    v = HRR.bytes_to_phases(r["hrr_vector"], dim=dim)
    a_score = HRR.similarity(HRR.unbind(v, key), HRR.bind(HRR.encode_text(r["content"], dim), rc))
    b_score = HRR.similarity(v, key)
    truth = entity.lower() in r["content"].lower()
    (hits if truth else misses).append((r["fact_id"], r["content"], a_score, b_score))

def summarize(label, idx):
    if not hits or not misses:
        print("%s: need both classes present (hits=%d misses=%d)" % (label, len(hits), len(misses)))
        return
    mh = sum(x[idx] for x in hits) / len(hits)
    mm = sum(x[idx] for x in misses) / len(misses)
    print("%s: matching mean %.3f | non-matching mean %.3f | separation %+.3f" % (label, mh, mm, mh - mm))

print("matching facts (text contains %r): %d | others: %d" % (entity, len(hits), len(misses)))
summarize("A plugin path (unbind+text)", 2)
summarize("B direct component       ", 3)

print("\ntop 5 by B (direct):")
for fid, content, a_s, b_s in sorted(hits + misses, key=lambda x: -x[3])[:5]:
    print("  B %.3f A %.3f truth=%s #%s %s" % (b_s, a_s, entity.lower() in content.lower(), fid, " ".join(content.split())[:90]))
print("\ntop 5 by A (plugin path):")
for fid, content, a_s, b_s in sorted(hits + misses, key=lambda x: -x[2])[:5]:
    print("  A %.3f B %.3f truth=%s #%s %s" % (a_s, b_s, entity.lower() in content.lower(), fid, " ".join(content.split())[:90]))
