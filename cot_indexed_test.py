# Experiment: does an answer list with column numbers (["Column 1: type", ...]) fix the wrong list lengths of CoT on wide tables?
# Same prompt as cta_cot.py (one call per table, no retry), only the answer format of the instructions and of the 4 examples changes.
# Only the tables with >= 21 columns of the evaluated datasets are used. Result: data/cot_examples is untouched, nothing is written but a csv.
import csv, json, re, sys
from concurrent.futures import ThreadPoolExecutor
import cta_cot as c

csv.field_size_limit(sys.maxsize)


def indexed_examples(dataset, setting):
    msgs = c.example_messages(dataset, setting)
    out = []
    for m in msgs:
        if m["role"] == "assistant":
            lst = re.findall(r"\[[^\[\]]*\]", m["content"])[-1]
            types = json.loads(lst)
            new = json.dumps([f"Column {i + 1}: {t}" for i, t in enumerate(types)], ensure_ascii=False)
            m = {"role": "assistant", "content": m["content"][:m["content"].rindex(lst)] + new + m["content"][m["content"].rindex(lst) + len(lst):]}
        out.append(m)
    return out


def system_msgs(labels):
    s = c.system_messages(labels, False)
    s[1]["content"] = s[1]["content"].replace('["type1", "type2", ...]', '["Column 1: type1", "Column 2: type2", ...]')
    return s


def parse_indexed(answer):
    for text in reversed(re.findall(r"\[[^\[\]]*\]", answer)):
        try:
            v = json.loads(text)
        except Exception:
            continue
        got = {}
        for e in v:
            m = re.match(r"\s*Column\s*(\d+)\s*:\s*(.*)$", str(e))
            if m:
                got[int(m.group(1))] = m.group(2).strip()
        if got:
            return got
    return None


def run(ds):
    tabs, labels = c.load_tables(ds, "test")
    tabs = [t for t in tabs if t["num_col"] >= 21 and any(len(g) == 1 for g in t["gold_labels"])]
    base = system_msgs(labels) + indexed_examples(ds, "single_label")

    def one(t):
        msgs, _ = c.fit_messages(base, t["table_raw"], t["num_col"])
        ans, pt, ct = c.chat(msgs)
        return t, parse_indexed(ans)

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, tabs))
    full = right = asked = n_ok = 0
    for t, got in res:
        n = t["num_col"]
        full += got is not None and all(i in got for i in range(1, n + 1))
        for ci, g in enumerate(t["gold_labels"]):
            if len(g) == 1:
                asked += 1
                p = c.resolve_to_label_set(got.get(ci + 1, ""), labels) if got else ""
                right += p == g[0]
    print(f"{ds}: wide tables {len(tabs)} | all column numbers answered {full}/{len(tabs)} | evaluated columns {asked}, correct {right} ({100*right/max(1,asked):.1f}%)", flush=True)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        run(d)
