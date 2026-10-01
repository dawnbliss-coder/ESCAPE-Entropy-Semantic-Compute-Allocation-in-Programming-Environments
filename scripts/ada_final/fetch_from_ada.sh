#!/bin/bash
# Ada -> laptop: the small final outputs only (scores, calibration, baselines, O5,
# fidelity rerun, logs, run registry, status). Not entropies or corpus bytes.
#   bash scripts/ada_final/fetch_from_ada.sh [remote_dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
REMOTE=${1:-escape-final-20260928}
mkdir -p results/final
rsync -a --info=stats1 "ada:~/$REMOTE/results/final/" results/final/
rsync -a "ada:~/$REMOTE/runs.json" results/final/runs.ada.json
echo "Fetched. Results are in results/final/ -- the mid-submission report's build pipeline was removed post-submission; see docs/FINAL_RUN.md to reconstruct it if rebuilding the report."
