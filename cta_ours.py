# -*- coding: utf-8 -*-
"""
Few-shot baseline of "Improving Column Type Annotation Using LLMs" (Babamahmoudi, Rafiei, Nascimento, VLDB 2025 TaDA) on the datasets
and test columns of the RACOON baseline, with our LLM (GLM-4-9B on the shared vLLM).

Single-column setting of the paper: the LLM sees ONE column (no other column of the table) and answers with one type.
Prompt = the one of the paper (cta.py, single_column_prompting): two system messages (candidate types, instructions), then the
5 examples as user/assistant turns, then the column to annotate. The 5 examples are drawn at random (fixed seed) from the TRAIN
split of the same dataset and are the same for every test column of the dataset.

Differences to the original code (cta.py): our datasets and their type sets; the column is shown as its first 6 cells (the project
rule, like RACOON) joined by a space; the answer is mapped to the type set with label_resolve (the same rule as for RACOON).

Usage: cta_ours.py <dataset> [--setting single_label|multi_label] [--shots 5] [--seed 0] [--max_columns N] [--shard K/N]
  dataset: sc | db | biodiv_db | biodiv_wiki
Result: outputs/<dataset>/fewshot_<setting>/results[.shardKofN].csv, 10 fields per row like RACOON_CTA.py, so that
RACOON_yu/scripts/eval/eval_racoon.py evaluates it unchanged.
"""
from __future__ import annotations

import argparse
import csv
import os
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

from openai import OpenAI

RACOON_SRC = os.environ.get("RACOON_SRC", "/nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/RAG/RACOON_yu/src")
sys.path.insert(0, RACOON_SRC)
from sc_db_loader import load_tables  # noqa: E402
from label_resolve import resolve_to_label_set  # noqa: E402

HERE = Path(__file__).resolve().parent
client = OpenAI(base_url=os.environ["VLLM_BASE_URL"], api_key=os.environ["VLLM_API_KEY"], timeout=300)
MODEL = os.environ["VLLM_MODEL"]


def columns_of(tables):
    """(table_id, column_id, cells, gold list) for every labeled column; cells = first 6 cells of that column."""
    out = []
    for t in tables:
        for ci, gold in enumerate(t["gold_labels"]):
            if gold:
                cells = [r[ci] for r in t["table_raw"] if r[ci] not in (None, "")]
                out.append((t["table_id"], t["col_ids"][ci], cells, gold))
    return out


def column_text(cells):
    return "Column 1: " + " ".join(cells)


def system_messages(labels, multi=False):
    if multi:  # multi-label setting: the only change to the paper's prompt is "one or multiple" (wording of RACOON appendix A.2)
        return [
            {"role": "system", "content": "Your task is to classify the columns of a given table with one or multiple of the following classes that are seperated with comma:"
                                          + ", ".join(labels) + "."},
            {"role": "system", "content": "Your instructions are: 1. Look at the column and the types given to you. 2. Examine the values of the column. "
                                          "3. Select one or multiple types that best represent the meaning of the column. 4. Answer with the selected types only, separated by ';', and print every type only once. "
                                          "The format of the answer should be like this: type1; type2  Print 'I don't know' if you are not able to find the semantic type."},
        ]
    return [
        {"role": "system", "content": "Your task is to classify the columns of a given table with only one of the following classes that are seperated with comma:"
                                      + ", ".join(labels) + "."},
        {"role": "system", "content": "Your instructions are: 1. Look at the column and the types given to you. 2. Examine the values of the column. "
                                      "3. Select a type that best represents the meaning of the column. 4. Answer with the selected type only, and print the type only once. "
                                      "The format of the answer should be like this: type  Print 'I don't know' if you are not able to find the semantic type."},
    ]


def mistral_embed(texts, model="mistral-embed"):
    """Embeddings from the Mistral API (same call as cta_glm.mistral_embed, which the original RAG code uses with OpenAI)."""
    texts = [t.replace("\n", " ")[:6000] or " " for t in texts]
    for attempt in range(20):
        try:
            req = urllib.request.Request("https://api.mistral.ai/v1/embeddings",
                data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["MISTRAL_API_KEY"]})
            with urllib.request.urlopen(req, timeout=120) as r:
                return [d["embedding"] for d in json.load(r)["data"]]
        except Exception:
            if attempt == 19:
                raise
            time.sleep(min(60, 5 * (attempt + 1)))


def embed_all(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        out += mistral_embed(texts[i:i + bs])
    return out


def ask(messages):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=MODEL, messages=messages, temperature=0, seed=42, max_tokens=200)
            m = r.choices[0].message
            return (m.content or getattr(m, "reasoning", None) or "").strip(), r.usage.prompt_tokens, r.usage.completion_tokens
        except Exception:
            if attempt == 2:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--setting", default="single_label", choices=["single_label", "multi_label"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_columns", type=int, default=None)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--method", default="fewshot", choices=["fewshot", "rag"])
    ap.add_argument("--pool", type=int, default=2000, help="rag: size of the reference sample of train columns (the paper uses 2000)")
    ap.add_argument("--build_pool_only", action="store_true")
    a = ap.parse_args()

    test_tables, labels = load_tables(a.dataset, "test")
    train_tables, _ = load_tables(a.dataset, "train")
    # single_label setting = the columns of the single_label folder (exactly one gold type); sc/db have only such columns
    multi = a.setting == "multi_label"
    ok_gold = (lambda g: len(g) >= 2) if multi else (lambda g: len(g) == 1)  # multi_label folder = >=2 golds, single_label folder = exactly 1
    test_cols = [c for c in columns_of(test_tables) if ok_gold(c[3])]
    # the examples: random single-gold columns of the train split (they must have cells)
    # no example may have the same cell content as a test column (BiodivTab, GitTables and TURL repeat columns across train/test),
    # and the examples differ from each other in content
    test_keys = {tuple(c[2]) for c in columns_of(test_tables)}
    pool = [c for c in columns_of(train_tables) if ok_gold(c[3]) and c[2] and tuple(c[2]) not in test_keys]
    rag = a.method == "rag"
    if rag:
        # reference pool = a random sample of `--pool` train columns (paper: 2000), distinct in content, none equal to a test column
        rng, uniq, seen = random.Random(a.seed), [], set()
        for c in rng.sample(pool, len(pool)):
            if tuple(c[2]) not in seen:
                seen.add(tuple(c[2]))
                uniq.append(c)
        ref = uniq[:a.pool]
        cache = HERE / "data" / "rag_cache" / f"{a.dataset}_{a.setting}_pool{a.pool}_seed{a.seed}.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            ref_vecs = json.load(open(cache))
            assert len(ref_vecs) == len(ref)
        else:
            ref_vecs = embed_all([" ".join(c[2]) for c in ref])
            json.dump(ref_vecs, open(cache, "w"))
        import numpy as np
        R = np.array(ref_vecs, dtype=np.float32)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        print(f"rag reference pool: {len(ref)} train columns (of {len(pool)} candidates), embeddings from {cache.name}", flush=True)
        if a.build_pool_only:
            return
        base = system_messages(labels, multi)
    else:
        rng, shots, seen = random.Random(a.seed), [], set()
        for c in rng.sample(pool, len(pool)):
            if tuple(c[2]) not in seen:
                seen.add(tuple(c[2]))
                shots.append(c)
            if len(shots) == a.shots:
                break
        print("few-shot examples:", [(s[3][0], " ".join(s[2])[:40]) for s in shots], flush=True)

        base = system_messages(labels, multi)
        for _, _, cells, gold in shots:
            base.append({"role": "user", "content": f"Classify this column:\n\n{column_text(cells)}"})
            base.append({"role": "assistant", "content": "; ".join(gold) if multi else gold[0]})

    k, n = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)
    todo = [c for i, c in enumerate(test_cols) if i % n == k]
    if a.max_columns:
        todo = todo[:a.max_columns]
    out_dir = HERE / "outputs" / a.dataset / f"{'rag' if rag else 'fewshot'}_{a.setting}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (f"results.shard{k}of{n}.csv" if a.shard else "results.csv")
    if rag:
        Q = np.array(embed_all([" ".join(c[2]) for c in todo]), dtype=np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for i, (tid, cid, cells, gold) in enumerate(todo):
            hint = ""
            if rag:
                sims = R @ Q[i]
                top = np.argsort(-sims)[:a.shots]  # most similar first, like the original code
                near = [ref[j] for j in top]
                assert all(tuple(c[2]) != tuple(cells) for c in near), "a retrieved example has the content of the test column"
                msgs = list(base)
                for c in near:
                    msgs.append({"role": "user", "content": f"Classify this column:\n\n{column_text(c[2])}"})
                    msgs.append({"role": "assistant", "content": "; ".join(c[3]) if multi else c[3][0]})
                hint = "|".join(f"{sims[j]:.3f}:{';'.join(ref[j][3])}" for j in top)
                msgs.append({"role": "user", "content": f"Classify this column:\n\n{column_text(cells)}"})
            else:
                msgs = base + [{"role": "user", "content": f"Classify this column:\n\n{column_text(cells)}"}]
            answer, pt, ct = ask(msgs)
            first = answer.splitlines()[0].strip() if answer else ""
            if multi:
                parts = [resolve_to_label_set(p.strip().strip("'\""), labels) for p in first.split(";")]
                pred = ";".join(dict.fromkeys(p for p in parts if p))
            else:
                pred = resolve_to_label_set(first.strip("'\""), labels)
            w.writerow([i, 0, pred, hint, answer, pt, ct, tid, cid, ";".join(gold)])
            fh.flush()
            if i % 50 == 0:
                print(f"{i}/{len(todo)}", flush=True)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
