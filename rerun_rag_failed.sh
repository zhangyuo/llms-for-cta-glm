#!/bin/bash
cd /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/LLM/llms-for-cta-glm
set -a; . /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/RAG/racoon_root/.env; . ~/RAG4CTA/.env; set +a
P=/nfs/srv/data2/af62lp@cs.aau.dk/miniconda3/envs/racoon/bin/python
for x in single_label:0 single_label:3 single_label:6 single_label:7 multi_label:4; do s=${x%%:*}; k=${x##*:}
  $P -u cta_ours.py turl --setting $s --method rag --shard $k/8 > log/rag_turl_${s}_shard${k}of8.log 2>&1 < /dev/null &
  sleep 20
done
wait
