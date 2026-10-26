#!/bin/bash
# CoT, multi_label setting: TURL in 8 table shards, BiodivTab-DB / Wiki in one process each
cd /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/LLM/llms-for-cta-glm
set -a; . /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/RAG/racoon_root/.env; set +a
P=/nfs/srv/data2/af62lp@cs.aau.dk/miniconda3/envs/racoon/bin/python
for d in biodiv_db biodiv_wiki; do $P -u cta_cot.py $d --setting multi_label > log/cot_${d}_multi_label.log 2>&1 < /dev/null & done
for k in 0 1 2 3 4 5 6 7; do $P -u cta_cot.py turl --setting multi_label --shard $k/8 > log/cot_turl_multi_label_shard${k}of8.log 2>&1 < /dev/null & done
wait
echo ALLDONE > log/cot_multi_ALLDONE
