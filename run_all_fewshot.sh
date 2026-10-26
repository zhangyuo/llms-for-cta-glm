#!/bin/bash
# few-shot on every dataset/setting; sharded where large
cd /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/LLM/llms-for-cta-glm
set -a; . /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/RAG/racoon_root/.env; set +a
P=/nfs/srv/data2/af62lp@cs.aau.dk/miniconda3/envs/racoon/bin/python
for j in "sc single_label" "db single_label" "biodiv_db single_label" "biodiv_db multi_label" "biodiv_wiki single_label" "biodiv_wiki multi_label"; do
  set -- $j; nohup $P -u cta_ours.py $1 --setting $2 > log/v2_fewshot_$1_$2.log 2>&1 < /dev/null &
done
for st in single_label multi_label; do for k in 0 1 2 3 4 5 6 7; do
  nohup $P -u cta_ours.py turl --setting $st --shard $k/8 > log/v2_fewshot_turl_${st}_shard${k}of8.log 2>&1 < /dev/null &
done; done
wait
echo ALLDONE > log/v2_fewshot_ALLDONE
