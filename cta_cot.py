# -*- coding: utf-8 -*-
"""
CoT baseline of "Improving Column Type Annotation Using LLMs" (Babamahmoudi, Rafiei, Nascimento, VLDB 2025 TaDA) on our datasets,
with our LLM (GLM-4-9B on the shared vLLM).

Multi-column setting of the paper: the LLM sees the WHOLE table and answers all its columns in one call; every example (4, from the
train split, data/cot_examples/, written by gen_cot_examples.py) has a reasoning step before the type list (Fig. 4 of the paper).
Prompt = the two system messages of the paper's multi-column prompt (candidate types, instructions with the list format), then the 4
examples as user/assistant turns, then the table to annotate ("Column i: first 6 cells").

Our additions (decided with the user):
  - every column of the table is shown, also the columns without a label (they are context); the LLM answers all of them and only
    the columns of the evaluated label setting are scored (single_label: exactly one gold type; multi_label: two or more);
  - a table whose prompt does not fit the context window (32,768 tokens) gets the TEXT OF ITS CELLS shortened (context_fit caps),
    no column or row is removed;
  - ANSWER FORMAT (the one change to the paper's prompt, decided with the user after a test on wide tables): the type list of the answer
    (and of the 4 examples) carries the column number, ["Column 1: type1", "Column 2: type2", ...], so that a model that skips or merges
    a column does not shift all following types. The paper's bare list gave a wrong list length for 50% of the tables with 21-40 columns
    and for all tables with more columns (0% for <= 10 columns);
  - one call per table, no retry, no repair: a column whose number is not in the answer gets an empty prediction (counted wrong); the
    entries are mapped to the type set with label_resolve (same rule as RACOON / few-shot / RAG);
  - tokens of a call are divided over the evaluated columns of the table (the metrics report tokens per column).

Usage: cta_cot.py <dataset> [--setting single_label] [--shard K/N] [--max_tables N]
Result: outputs/<dataset>/cot_<setting>/results[.shardKofN].csv, 10 fields per row like RACOON_CTA.py (evaluate with eval_racoon.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, os.environ.get("RACOON_SRC", "/nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/RAG/RACOON_yu/src"))
from sc_db_loader import load_tables  # noqa: E402
from label_resolve import resolve_to_label_set  # noqa: E402
from context_fit import CAPS, count_tokens  # noqa: E402

HERE = Path(__file__).resolve().parent
client = OpenAI(base_url=os.environ["VLLM_BASE_URL"], api_key=os.environ["VLLM_API_KEY"], timeout=600)
MODEL = os.environ["VLLM_MODEL"]
CONTEXT, ANSWER = 32768, 4000


def table_text(cols):
    return "\n".join(f"Column {i + 1}: {' '.join(c)}" for i, c in enumerate(cols))


def system_messages(labels, multi):
    many = "one or multiple" if multi else "only one"
    sep = "; " if multi else ""
    fmt = ("[\"Column 1: type1\", \"Column 2: type2\", ...]" if not multi
           else "[\"Column 1: type1; type2\", \"Column 2: type3\", ...] (several types of one column separated by ';')")
    return [
        {"role": "system", "content": f"Your task is to classify the columns of a given table with {many} of the following classes that are seperated with comma:"
                                      + ", ".join(labels) + "."},
        {"role": "system", "content": "Your instructions are: 1. Look at the columns and the types given to you. 2. Examine the values of the columns. "
                                      "3. Select a type that best represents the meaning of each column. 4. Answer with the selected type only. "
                                      f"The format of the answer should be like this: {fmt}. "
                                      "Print 'I don't know' if you are not able to find the semantic type."},
    ]


def example_messages(dataset, setting):
    d = json.load(open(HERE / "data" / "cot_examples" / f"{dataset}_{setting}.json"))
    assert d["status"] in ("auto_selected", "approved"), d["status"]
    msgs = []
    for e in d["examples"]:
        msgs.append({"role": "user", "content": "Classify these column:\n\n" + table_text(e["columns"])})
        r = e["reasoning"]
        lst = re.findall(r"\[[^\[\]]*\]", r)[-1]
        at = r.rindex(lst)
        new = json.dumps([f"Column {i + 1}: {t}" for i, t in enumerate(json.loads(lst))], ensure_ascii=False)
        msgs.append({"role": "assistant", "content": r[:at] + new + r[at + len(lst):]})
    return msgs


def chat(messages, temperature=0.0):
    r = client.chat.completions.create(model=MODEL, messages=messages, temperature=temperature, seed=42, max_tokens=ANSWER)
    m = r.choices[0].message
    return (m.content or getattr(m, "reasoning", None) or "").strip(), r.usage.prompt_tokens, r.usage.completion_tokens


def parse_list(answer):
    found = re.findall(r"\[[^\[\]]*\]", answer)
    for text in reversed(found):
        try:
            v = json.loads(text)
        except Exception:
            try:
                import ast
                v = ast.literal_eval(text)
            except Exception:
                continue
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return v
    return None


def parse_indexed(answer):
    """{column number: type text} from the last [...] list of the answer whose entries look like "Column k: type"; None when there is none."""
    for text in reversed(re.findall(r"\[[^\[\]]*\]", answer)):
        try:
            v = json.loads(text)
        except Exception:
            try:
                import ast
                v = ast.literal_eval(text)
            except Exception:
                continue
        got = {}
        for e in v if isinstance(v, list) else []:
            m = re.match(r"\s*Column\s*(\d+)\s*:\s*(.*)$", str(e))
            if m:
                got.setdefault(int(m.group(1)), m.group(2).strip())
        if got:
            return got
    return None


def build_columns(table_raw, num_col, cap):
    return [[(r[ci][:cap] if cap else r[ci]) for r in table_raw if r[ci]] for ci in range(num_col)]


def fit_messages(base, table_raw, num_col):
    """Shorten the cell text (never drop a column or a row) until the prompt fits; returns (messages, cap used)."""
    budget = CONTEXT - ANSWER - 500
    for cap in (None,) + CAPS:
        cols = build_columns(table_raw, num_col, cap)
        msgs = base + [{"role": "user", "content": "Classify these column:\n\n" + table_text(cols)}]
        chars = sum(len(m["content"]) for m in msgs)
        if chars < 30000 or count_tokens(" ".join(m["content"] for m in msgs)) <= budget:
            return msgs, cap
    return msgs, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--setting", default="single_label", choices=["single_label", "multi_label"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--max_tables", type=int, default=None)
    a = ap.parse_args()
    multi = a.setting == "multi_label"
    ok = (lambda g: len(g) >= 2) if multi else (lambda g: len(g) == 1)

    tables, labels = load_tables(a.dataset, "test")
    tables = [t for t in tables if any(ok(g) for g in t["gold_labels"])]  # tables with at least one evaluated column
    k, n = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)
    todo = [t for i, t in enumerate(tables) if i % n == k]
    if a.max_tables:
        todo = todo[:a.max_tables]
    base = system_messages(labels, multi) + example_messages(a.dataset, a.setting)
    out_dir = HERE / "outputs" / a.dataset / f"cot_{a.setting}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (f"results.shard{k}of{n}.csv" if a.shard else "results.csv")
    stats = {"cut": 0, "no_list": 0, "missing_numbers": 0}
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        row_no = 0
        for ti, t in enumerate(todo):
            msgs, cap = fit_messages(base, t["table_raw"], t["num_col"])
            stats["cut"] += cap is not None
            answer, pt, ct = chat(msgs)
            got = parse_indexed(answer)
            if got is None:
                stats["no_list"] += 1
            elif not all(i in got for i in range(1, t["num_col"] + 1)):
                stats["missing_numbers"] += 1
            asked = [ci for ci, g in enumerate(t["gold_labels"]) if ok(g)]
            for j, ci in enumerate(asked):
                txt = (got or {}).get(ci + 1)
                if txt is None:
                    pred, hint = "", "missing"
                elif multi:
                    parts = [resolve_to_label_set(p.strip(), labels) for p in txt.split(";")]
                    pred, hint = ";".join(dict.fromkeys(p for p in parts if p)), ""
                else:
                    pred, hint = resolve_to_label_set(txt, labels), ""
                w.writerow([row_no, 0, pred, hint, answer if j == 0 else "", round(pt / len(asked)), round(ct / len(asked)),
                            t["table_id"], t["col_ids"][ci], ";".join(t["gold_labels"][ci])])
                row_no += 1
            fh.flush()
            if ti % 25 == 0:
                print(f"{ti}/{len(todo)} tables | {stats}", flush=True)
    print("written", out, stats, flush=True)


if __name__ == "__main__":
    main()
