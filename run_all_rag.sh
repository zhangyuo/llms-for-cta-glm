#!/bin/bash
cd /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/LLM/llms-for-cta-glm
set -a; . /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/RAG/racoon_root/.env; . ~/RAG4CTA/.env; set +a
P=/nfs/srv/data2/af62lp@cs.aau.dk/miniconda3/envs/racoon/bin/python
J="sc:single_label db:single_label turl:single_label turl:multi_label biodiv_db:single_label biodiv_db:multi_label biodiv_wiki:single_label biodiv_wiki:multi_label"
for j in $J; do d=${j%%:*}; s=${j##*:}; $P -u cta_ours.py $d --setting $s --method rag --build_pool_only > log/rag_pool_${d}_$s.log 2>&1 & done
wait
for j in $J; do d=${j%%:*}; s=${j##*:}
  if [ $d = turl ]; then for k in 0 1 2 3 4 5 6 7; do $P -u cta_ours.py $d --setting $s --method rag --shard $k/8 > log/rag_${d}_${s}_shard${k}of8.log 2>&1 < /dev/null & done
  else $P -u cta_ours.py $d --setting $s --method rag > log/rag_${d}_$s.log 2>&1 < /dev/null & fi
done
wait
echo ALLDONE > log/rag_ALLDONE
