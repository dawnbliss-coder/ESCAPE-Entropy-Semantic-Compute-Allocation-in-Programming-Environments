#!/bin/bash
# Full-run Step 1 (docs/ADA_FULL_RUN_GATES.md): golden gate for escape_distill.extract.
# Exits non-zero unless the committed golden table is reproduced (F1/P/R within +-0.002),
# so the full-corpus student extraction can be chained on it with afterok.
#SBATCH --job-name=esc-golden-gate
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=results/final/logs/golden_gate-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/ada_final/env.sh
nvidia-smi -L
a_py -m escape_distill.extract --input golden --domains py cpp prose \
  --model student --run-id student-golden-gate --device cuda --batch-windows 64
a_py -m escape_distill.score_a_runs --run-ids student-golden-gate --golden --out results/distill/golden_gate_ada.json
a_py - <<'PY'
import json, sys
got = json.load(open("results/distill/golden_gate_ada.json"))["domains"]
ref = json.load(open("results/distill/golden_gate_local.json"))["domains"]
expected = {  # Gate 1: (F1, P, R) for the student, F1 for the strongest baseline
    "py": {"student-golden-gate": (0.1313, 0.0716, 0.7887), "baseline_indent_change": (0.4771,)},
    "cpp": {"student-golden-gate": (0.2196, 0.1264, 0.8364), "baseline_word_start": (0.3869,)},
    "prose": {"student-golden-gate": (0.4290, 0.2756, 0.9677), "baseline_word_start": (0.7601,)},
}
ok = True
for dom, rows in expected.items():
    for name, vals in rows.items():
        r = got[dom]["rows"][name]
        obs = (r["f1"], r["precision"], r["recall"])[:len(vals)]
        good = all(abs(o - e) <= 0.002 for o, e in zip(obs, vals))
        ok &= good
        print(f"{dom:6s} {name:24s} obs={' '.join(f'{o:.4f}' for o in obs)} "
              f"exp={' '.join(f'{e:.4f}' for e in vals)} {'PASS' if good else 'FAIL'}")
    s, l = got[dom]["rows"]["student-golden-gate"], ref[dom]["rows"]["student-golden-gate"]
    print(f"{dom:6s} tp/fp/fn ada={s['tp']}/{s['fp']}/{s['fn']} local={l['tp']}/{l['fp']}/{l['fn']}")
print("GATE 1", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
rm -rf "$SCR"
