# Recompute the reference vectors of the repo with Mistral's embedding model (same texts, same file layout as the originals)
import os, json, sys
import pandas as pd
sys.argv = ["x"]
os.environ.setdefault("VLLM_BASE_URL", "x"); os.environ.setdefault("VLLM_API_KEY", "x"); os.environ.setdefault("VLLM_MODEL", "x")
import cta_glm as c
def embed_all(texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        out += c.mistral_embed(texts[i:i + bs]); print(f"  {len(out)}/{len(texts)}", flush=True)
    return out
r = pd.read_csv("data/RAG_2000_sample.csv")
r["embedding"] = [str(v) for v in embed_all(r["data"].astype(str).tolist())]
r.to_csv("data/RAG_2000_sample_mistral.csv", index=False)
m = pd.read_csv("data/trainset_for_multi_rag_embeds.csv")
m["embedding"] = [str(v) for v in embed_all(m["formatted_string"].astype(str).tolist())]
m.to_csv("data/trainset_for_multi_rag_embeds_mistral.csv", index=False)
print("done:", len(r), "single-column references,", len(m), "table references")
