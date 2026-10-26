# -*- coding: utf-8 -*-
"""
Builds the CoT examples (multi-column setting) of the "Improving Column Type Annotation Using LLMs" baseline for OUR datasets.

The paper (Sec. 3.4, Fig. 4) adds a reasoning step before the answer of every few-shot example: the reasoning uses the values of the
columns and the relation between the columns to say what each column is, then gives the type list. The original code has 4 hand-written
examples for VizNet. Here GLM-4-9B writes the reasoning for example tables of the TRAIN split, given the table and the TRUE type of
every column (the same situation as an author who writes the reasoning knowing the answer).

Leakage rules (the examples must never contain a test column):
  - the example table is not a test table;
  - no column of the example table has the same first-6-cell content as ANY column of ANY test table;
  - optional (--jaccard 0.5, OFF by default = the rule of the other baselines, as in the original code which has no filter at all):
    no column of the example table shares >= that fraction of its cells with any test column.
Further rules: only the labeled columns of a train table are shown in the example (exactly one gold type in the single_label setting, >= 2 in the
multi_label setting; columns without a true type are left out), 3 to --max_cols such columns, the 4 examples differ in content and in their type lists.
The generated reasoning is checked by program: the type list at its end must equal the true types of the columns, in order; otherwise it
is generated again (max 6 attempts).

Selection: --n_cand (default 12) candidate examples are generated, cheap rules drop unsuitable tables (long-text cells, a single type for
all columns), then every candidate gets (a) a program check: the share of the quoted values of the reasoning that really occur in the
table, and (b) a GLM judge score (1-5 each for: cites concrete cells, uses the other columns as context, consistent with the cells).
The --n best candidates with different type lists are kept. Human review is optional.

Usage: gen_cot_examples.py <dataset> [--setting single_label] [--n 4] [--n_cand 12] [--seed 0] [--max_cols 6]
Result: data/cot_examples/<dataset>_<setting>.json (status "auto_selected") and <dataset>_<setting>.candidates.json (all candidates, scores)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, os.environ.get("RACOON_SRC", "/nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/RAG/RACOON_yu/src"))
from sc_db_loader import load_tables  # noqa: E402

HERE = Path(__file__).resolve().parent
client = OpenAI(base_url=os.environ["VLLM_BASE_URL"], api_key=os.environ["VLLM_API_KEY"], timeout=300)
MODEL = os.environ["VLLM_MODEL"]

STYLE = '''Example of the style (the answer is known, the reasoning explains it):
Column 1: Australia Brazil France Germany Italy
Column 2: WIN WIN LOST WIN LOST
Column 3: AUS BRA FRA GER ITA
Reasoning: Based on the names of countries in the first column and the presence of results like "WIN" and "LOST" in the second column, it appears that the first column represents teams, the second column indicates the result of a match, and the third column contains the corresponding team symbols. Therefore, the column types are ["team", "result", "symbol"].'''


def table_text(cols):
    return "\n".join(f"Column {i + 1}: {' '.join(c)}" for i, c in enumerate(cols))


def writer_prompt(cols, types, multi=False):
    extra = (" Every column has one or several true types; an entry of the list joins the several types of one column with '; ' "
             '(for example "type a; type b"), exactly as given, and the reasoning explains every type of the column.') if multi else ""
    return (f"{STYLE}\n\nNow write the reasoning for the table below in the same style. The true semantic type of every column is given. "
            "Explain WHY each column has its type by quoting concrete cell values and by using the other columns of the table as context. "
            "Write one paragraph that starts with 'Reasoning:' and ends with the sentence 'Therefore, the column types are [...]' "
            "which lists exactly the given types, in the column order, each in double quotes. Do not add anything after that sentence." + extra + "\n\n"
            f"{table_text(cols)}\n\nTrue types in column order: {json.dumps(types, ensure_ascii=False)}")


def ask(prompt):
    r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=700)
    m = r.choices[0].message
    return (m.content or getattr(m, "reasoning", None) or "").strip()


def final_list(text):
    found = re.findall(r"column types are\s*(\[[^\]]*\])", text)
    if not found:
        return None
    try:
        return json.loads(found[-1])
    except Exception:
        return None


def quoted_share(reasoning, cols):
    """Share of the double-quoted values of the reasoning (without the final type list) that occur in the table."""
    body = reasoning.split("column types are")[0]
    vals = [v.strip(" ,.") for v in re.findall(r'"([^"]{2,80})"', body)]
    if not vals:
        return 0.0, 0
    text = " ".join(" ".join(c) for c in cols).lower()
    return sum(v.lower() in text for v in vals) / len(vals), len(vals)


def judge(cols, types, reasoning):
    prompt = ("Rate the reasoning below for a column type annotation example. The table and the true column types are given.\n"
              "Score each criterion from 1 (bad) to 5 (very good):\n"
              "cites_cells: it quotes concrete cell values of the table;\n"
              "uses_other_columns: it uses the other columns of the table as context to decide what a column is;\n"
              "consistent: every statement agrees with the cell values (no invented facts, no wrong claims).\n"
              'Answer with JSON only, like {"cites_cells": 4, "uses_other_columns": 3, "consistent": 5}.\n\n'
              f"{table_text(cols)}\n\nTrue types: {json.dumps(types, ensure_ascii=False)}\n\n{reasoning}")
    for _ in range(3):
        r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=200)
        m = r.choices[0].message
        t = (m.content or getattr(m, "reasoning", None) or "")
        f = re.search(r"\{[^{}]*\}", t)
        try:
            d = json.loads(f.group(0))
            return {k: int(d[k]) for k in ("cites_cells", "uses_other_columns", "consistent")}
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--setting", default="single_label", choices=["single_label", "multi_label"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--n_cand", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_cols", type=int, default=3)
    ap.add_argument("--max_cols", type=int, default=6)
    ap.add_argument("--jaccard", type=float, default=0.0, help="0 = only exact-content test columns are excluded (default)")
    a = ap.parse_args()
    multi = a.setting == "multi_label"
    ok = (lambda g: len(g) >= 2) if multi else (lambda g: len(g) == 1)

    test_tables, _ = load_tables(a.dataset, "test")
    train_tables, _ = load_tables(a.dataset, "train")
    test_ids = {t["table_id"] for t in test_tables}
    test_keys, inv, test_sets = set(), collections.defaultdict(set), []
    for t in test_tables:  # ALL columns of ALL test tables, labeled or not
        for ci in range(t["num_col"]):
            cells = [r[ci] for r in t["table_raw"] if r[ci]]
            test_keys.add(tuple(cells))
            for x in set(cells):
                inv[x].add(len(test_sets))
            test_sets.append(set(cells))

    def leaks(cells):
        if tuple(cells) in test_keys:
            return True
        if a.jaccard <= 0:
            return False
        s = set(cells)
        near = collections.Counter(j for x in s for j in inv.get(x, ()))
        return any(len(s & test_sets[j]) / len(s | test_sets[j]) >= a.jaccard for j in near)

    cand = []
    for t in train_tables:
        if t["table_id"] in test_ids:
            continue
        # only the labeled columns are shown in an example (a column without a true type cannot be part of the answer)
        keep = [ci for ci in range(t["num_col"]) if ok(t["gold_labels"][ci])]
        cols = [[r[ci] for r in t["table_raw"] if r[ci]] for ci in keep]
        if not (a.min_cols <= len(keep) <= a.max_cols) or any(not c for c in cols):
            continue
        cand.append((dict(t, gold_labels=[t["gold_labels"][ci] for ci in keep], num_col=len(keep)), cols))
    print(f"{a.dataset} {a.setting}: {len(cand)} candidate train tables ({a.min_cols}-{a.max_cols} labeled columns, unlabeled columns left out)", flush=True)
    random.Random(a.seed).shuffle(cand)

    chosen, seen_content, rejected, tried = [], set(), 0, 0
    for t, cols in cand:
        types = ["; ".join(g) for g in t["gold_labels"]]
        key = tuple(map(tuple, cols))
        if key in seen_content or len(set(types)) < 2:
            continue
        if max(sum(len(c) for c in col) / len(col) for col in cols) > 80:  # long-text cells make a poor, long example
            continue
        if any(leaks(c) for c in cols):
            rejected += 1
            continue
        tried += 1
        if tried > 4 * a.n_cand:
            break
        reasoning = None
        for attempt in range(6):
            text = ask(writer_prompt(cols, types, multi))
            if final_list(text) == types and text.lstrip().startswith("Reasoning:") and text.rstrip().split("]")[-1].strip() in ("", "."):
                reasoning = text.strip()
                break
        if reasoning is None:
            continue
        share, nq = quoted_share(reasoning, cols)
        if nq < 2 or share < 0.7:
            continue
        sc = judge(cols, types, reasoning)
        if sc is None:
            continue
        seen_content.add(key)
        chosen.append({"table_id": t["table_id"], "columns": cols, "types": types, "reasoning": reasoning,
                       "quoted_share": round(share, 2), "n_quoted": nq, "judge": sc, "score": sum(sc.values()) + share})
        print(f"  candidate {len(chosen)}: {t['table_id'][:30]} | judge {sc} | quoted in table {share:.2f} | types {types}", flush=True)
        if len(chosen) == a.n_cand:
            break
    print(f"leaking candidates skipped: {rejected}; candidates kept: {len(chosen)}", flush=True)
    best, seen_types = [], set()
    for e in sorted(chosen, key=lambda e: -e["score"]):
        if tuple(e["types"]) not in seen_types:
            seen_types.add(tuple(e["types"]))
            best.append(e)
        if len(best) == a.n:
            break
    if len(best) < a.n:  # small train split: fill up with the next best candidates (other tables, but a type list that was used already)
        for e in sorted(chosen, key=lambda e: -e["score"]):
            if e not in best:
                best.append(e)
            if len(best) == a.n:
                break
        print(f"filled up with candidates of repeated type lists: {len(best)} examples", flush=True)
    if not best:
        sys.exit("no example found")
    if len(best) < a.n:
        print(f"WARNING: only {len(best)} examples exist (small train split), the prompt gets {len(best)}", flush=True)
    cand_out = HERE / "data" / "cot_examples" / f"{a.dataset}_{a.setting}.candidates.json"
    cand_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(chosen, open(cand_out, "w"), ensure_ascii=False, indent=1)
    chosen = best
    out = HERE / "data" / "cot_examples" / f"{a.dataset}_{a.setting}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"status": "auto_selected", "dataset": a.dataset, "setting": a.setting, "seed": a.seed, "examples": chosen}, open(out, "w"), ensure_ascii=False, indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
