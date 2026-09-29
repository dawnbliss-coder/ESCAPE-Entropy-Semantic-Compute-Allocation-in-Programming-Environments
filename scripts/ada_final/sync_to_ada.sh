#!/bin/bash
# Laptop -> Ada. Copies code, manifest, schemas and small committed results into a NEW
# remote directory; never touches existing Ada projects. Bulky regenerable data
# (corpus bytes, entropies, venvs) is rebuilt on Ada by the jobs, so it is excluded.
#   bash scripts/ada_final/sync_to_ada.sh [remote_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
REMOTE=${1:-escape-final-20260928}
ssh ada "mkdir -p ~/$REMOTE"
rsync -a --info=stats1 \
  --exclude '.venv*' --exclude '.deps-*' --exclude '.c-deps' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'checkpoints/' --exclude 'entropies/' --exclude 'boundaries/' --exclude 'structure/' \
  --exclude 'whitespace/' --exclude 'identifiers/' --exclude 'memory_regions/' --exclude 'analysis_regions/' \
  --exclude '/corpus/py/' --exclude '/corpus/cpp/' --exclude '/corpus/prose/' \
  --exclude 'reports/*/build/' --exclude '.claude/' --exclude 'results/final/' \
  ./ "ada:~/$REMOTE/"
echo "Synced to ada:~/$REMOTE. Next, on Ada:"
echo "  cd ~/$REMOTE && bash scripts/ada_final/submit_all.sh"
