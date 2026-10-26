#!/bin/bash
cd /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/LLM/llms-for-cta-glm
set -a; . /nfs/srv/data2/af62lp@cs.aau.dk/cta_baselines/RAG/racoon_root/.env; . ~/RAG4CTA/.env; set +a
exec /nfs/srv/data2/af62lp@cs.aau.dk/miniconda3/envs/racoon/bin/python -u cta_ours.py "$@"
