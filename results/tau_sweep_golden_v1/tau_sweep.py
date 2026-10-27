"""Tau-sensitivity sweep on the golden fixtures (teacher_native + student).

Read-only w.r.t. the repo. Recomputes per-byte entropies once per file/stream
(teacher native swa512; student stride-1 rolling 512-byte windows), saves them
under OUT/entropies_*.npz, then re-thresholds at TAU_GRID and scores every point
against the same golden targets with the fixed-EOF null (R=10,000), k=0, and the
three structure-blind baselines (baselines do not depend on tau).

Memory-safe settings: batch 4 windows, 4 torch threads, MALLOC_ARENA_MAX=2,
ulimit -v 8GB.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/shashwat-11/ESCAPE/ESCAPE-Entropy-Semantic-Compute-Allocation-in-Programming-Environments")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from escape_distill.student_blt import StudentBLT, StudentConfig  # noqa: E402
from escape_scoring.baselines import indent_change, newline, word_start  # noqa: E402
from escape_scoring.metrics import match, rng_for, shuffled_boundaries  # noqa: E402
from stream_a.patching import bytes_to_token_ids  # noqa: E402
from run_golden_eval import (banded_mask, c_v1_patches, interior_starts,  # noqa: E402
                             load_teacher, per_byte_entropy, targets_from_parquet)  # noqa: E402

TAU_GRID = [0.8, 1.0, 1.335442066192627, 1.7, 2.0, 2.5]
SEED = 20260927
N_PERMS = 10000
GOLDEN = REPO / "golden_fixtures"
OUT = REPO / "results/tau_sweep_golden_v1"


def teacher_native_entropy(teacher, content: bytes) -> np.ndarray:
    ids = torch.from_numpy(bytes_to_token_ids(content)).unsqueeze(0)
    logits = teacher.logits(ids, mask=banded_mask(ids.shape[1])).float()
    logp = torch.log_softmax(logits, -1)
    ent = -(logp.exp() * logp).sum(-1)[0].numpy()
    return ent[: len(content)].astype(np.float32)


def main():
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    teacher, pc = load_teacher()
    ckpt = torch.load(HERE / "student.pt", map_location="cpu", weights_only=False)
    student = StudentBLT(StudentConfig(**ckpt["config"]))
    student.load_state_dict(ckpt["model_state"])
    student.eval()

    domains = ("py", "cpp", "prose")
    files = {d: sorted((GOLDEN / d).glob("*.bin")) for d in domains}

    config = {
        "experiment": "tau_sweep_golden_v1",
        "streams": {"teacher_native": "swa512 banded mask over [BOS]+file (reference backend, float32 cpu)",
                    "student": "stride-1 fresh windows <=512B (trained regime)"},
        "tau_grid": TAU_GRID, "k": 0, "seed": SEED, "n_permutations": N_PERMS,
        "teacher": {"checkpoint": "itazap/blt-1b-hf",
                    "revision": "91aa6b8e168046ad91e517d2191e1e974f50cc01",
                    "patcher_weights_sha256": hashlib.sha256(
                        (REPO / "checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01"
                         / "patcher.safetensors").read_bytes()).hexdigest()},
        "student": {"params": student.num_parameters(), "steps": ckpt["steps"],
                    "student_pt_sha256": hashlib.sha256((HERE / "student.pt").read_bytes()).hexdigest()},
        "torch": torch.__version__, "device": "cpu",
        "note": "golden-fixture variant; null recomputed per (stream, file, tau) with 10,000 fixed-EOF permutations",
    }

    # ---- 1. entropy arrays (once per file/stream) ----
    for d in domains:
        npz_t, npz_s = {}, {}
        for p in files[d]:
            fid, content = p.stem, p.read_bytes()
            t0 = time.time()
            e_t = teacher_native_entropy(teacher, content)
            e_s, _, dt_s = per_byte_entropy(lambda b: student(b), content)
            npz_t[fid], npz_s[fid] = e_t, e_s
            print(f"[entropy] {d}/{fid}: {len(content)}B teacher={time.time()-t0:.1f}s "
                  f"student={dt_s:.1f}s", flush=True)
        np.savez_compressed(OUT / f"entropies_teacher_native_{d}.npz", **npz_t)
        np.savez_compressed(OUT / f"entropies_student_{d}.npz", **npz_s)
    print("[entropy] saved", flush=True)

    # ---- 2. sweep ----
    results = {"config": config, "domains": {}}
    for d in domains:
        dom = {"tau_rows": [], "best_baseline": None}
        fid_order = [p.stem for p in files[d]]
        npz_t = np.load(OUT / f"entropies_teacher_native_{d}.npz")
        npz_s = np.load(OUT / f"entropies_student_{d}.npz")
        tgt, base = {}, {}
        for fid in fid_order:
            content = (GOLDEN / d / f"{fid}.bin").read_bytes()
            n = len(content)
            tgt[fid] = targets_from_parquet(GOLDEN / d / f"{fid}.structure.parquet", n)
            base[fid] = {b: sorted({int(o) for o in f(content) if 0 < o < n})
                         for b, f in (("newline", newline), ("indent_change", indent_change),
                                      ("word_start", word_start))}
        base_rows = {}
        for b in ("newline", "indent_change", "word_start"):
            tp = sum(len(match(base[fid][b], tgt[fid], 0)) for fid in fid_order)
            fp = sum(len(base[fid][b]) for fid in fid_order) - tp
            fn = sum(len(tgt[fid]) for fid in fid_order) - tp
            base_rows[b] = {"tp": tp, "fp": fp, "fn": fn,
                            "f1": 2 * tp / max(1e-12, 2 * tp + fp + fn)}
        best_base = max(base_rows, key=lambda b: base_rows[b]["f1"])
        dom["best_baseline"] = {"name": best_base, "f1": base_rows[best_base]["f1"]}

        for tau in TAU_GRID:
            row = {"tau": tau}
            for s, npz in (("teacher_native", npz_t), ("student", npz_s)):
                patches, pred, tp, fp, fn = {}, {}, 0, 0, 0
                for fid in fid_order:
                    e = npz[fid]
                    patches[fid] = c_v1_patches(e, tau)
                    pred[fid] = interior_starts(e, tau)
                    t = len(match(pred[fid], tgt[fid], 0))
                    tp += t; fp += len(pred[fid]) - t; fn += len(tgt[fid]) - t
                obs_f1 = 2 * tp / max(1e-12, 2 * tp + fp + fn)
                rngs = {fid: rng_for(SEED, fid, f"tau_sweep_{s}") for fid in fid_order}
                null_f1 = np.empty(N_PERMS)
                for r in range(N_PERMS):
                    t_all = f_all = n_all = 0
                    for fid in fid_order:
                        nb = shuffled_boundaries(patches[fid], len(npz[fid]), rngs[fid],
                                                 eof_policy="fixed_eof")
                        t = len(match(nb, tgt[fid], 0))
                        t_all += t; f_all += len(nb) - t; n_all += len(tgt[fid]) - t
                    null_f1[r] = 2 * t_all / max(1e-12, 2 * t_all + f_all + n_all)
                row[s] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": obs_f1,
                          "f1_h0_mean": float(null_f1.mean()),
                          "f1_margin": float(obs_f1 - null_f1.mean()),
                          "f1_permutation_p": float((1 + np.sum(null_f1 >= obs_f1)) / (N_PERMS + 1)),
                          "n_boundaries": tp + fp,
                          "boundary_density": (tp + fp) / sum(len(npz[f]) for f in fid_order),
                          "f1_minus_best_baseline": float(obs_f1 - base_rows[best_base]["f1"])}
            dom["tau_rows"].append(row)
            print(f"[sweep] {d} tau={tau:.3f}  "
                  f"teacher F1={row['teacher_native']['f1']:.4f} margin={row['teacher_native']['f1_margin']:+.4f} "
                  f"student F1={row['student']['f1']:.4f} margin={row['student']['f1_margin']:+.4f}", flush=True)
        results["domains"][d] = dom

    (OUT / "tau_sweep_scores.json").write_text(json.dumps(results, indent=2))
    print(f"\n[wrote] {OUT / 'tau_sweep_scores.json'}")

    print("\n=== tau sweep summary (golden, k=0) ===")
    for d in domains:
        dom = results["domains"][d]
        print(f"\n{d}  (best baseline: {dom['best_baseline']['name']} F1={dom['best_baseline']['f1']:.3f})")
        print(f"  {'tau':>7} | {'teacher F1':>10} {'margin':>8} {'P':>6} {'R':>6} {'bnd':>5} | "
              f"{'student F1':>10} {'margin':>8} {'P':>6} {'R':>6} {'bnd':>5}")
        for row in dom["tau_rows"]:
            t, s = row["teacher_native"], row["student"]
            print(f"  {row['tau']:7.3f} | {t['f1']:10.4f} {t['f1_margin']:+8.4f} {t['precision']:6.3f} "
                  f"{t['recall']:6.3f} {t['n_boundaries']:5d} | {s['f1']:10.4f} {s['f1_margin']:+8.4f} "
                  f"{s['precision']:6.3f} {s['recall']:6.3f} {s['n_boundaries']:5d}")


if __name__ == "__main__":
    main()
