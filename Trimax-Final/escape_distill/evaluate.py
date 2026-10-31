"""Evaluate a distilled StudentBLT against the BLT teacher on held-out windows.

Everything is computed on <=512-byte windows, where the teacher's swa512 regime equals
full causal attention, so the student (512-byte receptive field) and teacher are compared
on exactly the same context.

Metrics, per domain and overall:
  * entropy agreement   -- MAE and Pearson r between per-byte teacher/student entropy (nats)
  * KL(teacher||student)-- mean per-byte, the distillation objective on held-out data
  * bits/byte           -- each model's own next-byte cross-entropy against the TRUE byte
  * boundary F1         -- derive patch-start offsets from each model's entropy with the
                           frozen tau and stream_a.patching.compute_patches, then score the
                           student's interior boundaries against the teacher's (k=0).
  * word-start F1       -- a structure-blind comparator: every word start in the window
                           (escape_scoring.baselines.word_start) scored against the
                           teacher's boundaries the same way, so the student's agreement
                           can be read against what plain word segmentation already gets.

An untrained student (same architecture, random init) is evaluated the same way as a
baseline, so the result shows how much the distillation actually moved.
"""

from __future__ import annotations

import argparse
import json
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from escape_distill.student_blt import BYTE_TOKEN_OFFSET, StudentBLT, StudentConfig, entropy_nats
from escape_scoring.baselines import word_start
from escape_distill.data import load_samples, build_windows, collate
from escape_distill.teacher import load_teacher, teacher_logits
from stream_a.patching import compute_patches

TAU = 1.335442066192627


def interior_starts(byte_entropies: np.ndarray, tau: float) -> set[int]:
    """Entropy-triggered patch starts at interior byte offsets (exclude 0)."""
    patches = compute_patches(byte_entropies, tau)
    return {int(s) for s in patches.starts if 0 < int(s) < len(byte_entropies)}


def prf(pred: set[int], target: set[int]) -> tuple[int, int, int]:
    tp = len(pred & target)
    return tp, len(pred) - tp, len(target) - tp  # tp, fp, fn


@torch.inference_mode()
def logits_for_window(model, ids: np.ndarray, device, is_teacher: bool):
    batch = collate([ids], device)
    if is_teacher:
        lg = teacher_logits(model, batch["input_ids"])[0]
    else:
        lg = model(batch["input_ids"])[0]
    return lg  # [L, V]


def eval_model_vs_teacher(student, teacher, windows, meta, device, tau=TAU):
    agg = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "ws_tp": 0, "ws_fp": 0, "ws_fn": 0,
                               "ent_abs": 0.0, "ent_n": 0,
                               "kl_sum": 0.0, "bpb_student": 0.0, "bpb_teacher": 0.0,
                               "t_ent": [], "s_ent": []})
    for ids, dom in zip(windows, meta):
        k = len(ids) - 1  # real bytes
        t_lg = logits_for_window(teacher, ids, device, True)[:k].float()
        s_lg = logits_for_window(student, ids, device, False)[:k].float()
        true_next = torch.from_numpy(ids[1:k + 1]).to(device)  # bytes b0..b_{k-1}

        t_ent = entropy_nats(t_lg).cpu().numpy()
        s_ent = entropy_nats(s_lg).cpu().numpy()
        t_logp = F.log_softmax(t_lg, -1)
        s_logp = F.log_softmax(s_lg, -1)
        kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)              # KL(teacher||student) per byte
        bpb_t = F.nll_loss(t_logp, true_next, reduction="sum").item() / np.log(2)
        bpb_s = F.nll_loss(s_logp, true_next, reduction="sum").item() / np.log(2)

        t_starts = interior_starts(t_ent, tau)
        s_starts = interior_starts(s_ent, tau)
        tp, fp, fn = prf(s_starts, t_starts)
        content = (ids[1:k + 1] - BYTE_TOKEN_OFFSET).astype(np.uint8).tobytes()
        ws_tp, ws_fp, ws_fn = prf({o for o in word_start(content) if 0 < o < k}, t_starts)

        for key in (dom, "all"):
            a = agg[key]
            a["tp"] += tp; a["fp"] += fp; a["fn"] += fn
            a["ws_tp"] += ws_tp; a["ws_fp"] += ws_fp; a["ws_fn"] += ws_fn
            a["ent_abs"] += float(np.abs(t_ent - s_ent).sum()); a["ent_n"] += k
            a["kl_sum"] += float(kl.sum().item())
            a["bpb_student"] += bpb_s; a["bpb_teacher"] += bpb_t
            a["t_ent"].append(t_ent); a["s_ent"].append(s_ent)

    rows = {}
    for key, a in agg.items():
        n = max(1, a["ent_n"])
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        te = np.concatenate(a["t_ent"]); se = np.concatenate(a["s_ent"])
        pear = float(np.corrcoef(te, se)[0, 1]) if len(te) > 2 and se.std() > 0 else 0.0
        rows[key] = {
            "n_bytes": n, "boundary_tp": tp, "boundary_fp": fp, "boundary_fn": fn,
            "boundary_precision": round(prec, 4), "boundary_recall": round(rec, 4),
            "boundary_f1": round(f1, 4),
            "word_start_vs_teacher_f1": round(2 * a["ws_tp"] / max(1, 2 * a["ws_tp"] + a["ws_fp"] + a["ws_fn"]), 4),
            "entropy_mae_nats": round(a["ent_abs"] / n, 4),
            "entropy_pearson_r": round(pear, 4),
            "kl_teacher_student_nats_per_byte": round(a["kl_sum"] / n, 4),
            "bits_per_byte_student": round(a["bpb_student"] / n, 4),
            "bits_per_byte_teacher": round(a["bpb_teacher"] / n, 4),
        }
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate a distilled student against the BLT teacher.")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--student", required=True, help="student.pt from train.py")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--domains", default=None)
    ap.add_argument("--teacher-dtype", default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    domains = tuple(args.domains.split(",")) if args.domains else None

    samples = load_samples(args.data, split=args.eval_split, limit=args.limit_files,
                           domains=domains, seed=args.seed)
    windows = build_windows(samples)
    meta = [s.domain for s in samples for _ in build_windows([s])]
    if not windows:
        raise SystemExit("no eval windows found")

    ckpt = torch.load(args.student, map_location=device, weights_only=False)
    cfg = StudentConfig(**ckpt["config"])
    student = StudentBLT(cfg).to(device); student.load_state_dict(ckpt["model_state"]); student.eval()
    torch.manual_seed(args.seed)  # reproducible untrained baseline (was unseeded before 2026-10-29)
    untrained = StudentBLT(cfg).to(device); untrained.eval()
    teacher = load_teacher(args.ckpt_dir, device=str(device), dtype=args.teacher_dtype)

    print(f"[eval] files={len(samples)} windows={len(windows)}")
    trained_rows = eval_model_vs_teacher(student, teacher, windows, meta, device)
    base_rows = eval_model_vs_teacher(untrained, teacher, windows, meta, device)

    result = {"tau": TAU, "n_files": len(samples), "n_windows": len(windows),
              "student_config": ckpt["config"], "student_params": student.num_parameters(),
              "trained": trained_rows, "untrained_baseline": base_rows}
    (out / "distill_eval.json").write_text(json.dumps(result, indent=2))

    with open(out / "distill_eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "n_bytes", "boundary_f1", "boundary_precision",
                    "boundary_recall", "entropy_mae_nats", "entropy_pearson_r",
                    "kl_teacher_student_nats_per_byte", "bits_per_byte_student", "bits_per_byte_teacher"])
        for model_name, rows in (("trained", trained_rows), ("untrained", base_rows)):
            for dom, r in sorted(rows.items()):
                w.writerow([model_name, dom, r["n_bytes"], r["boundary_f1"], r["boundary_precision"],
                            r["boundary_recall"], r["entropy_mae_nats"], r["entropy_pearson_r"],
                            r["kl_teacher_student_nats_per_byte"], r["bits_per_byte_student"],
                            r["bits_per_byte_teacher"]])

    print("\n=== Student vs teacher (overall) ===")
    t, b = trained_rows["all"], base_rows["all"]
    print(f"  boundary F1        trained {t['boundary_f1']:.3f}   untrained {b['boundary_f1']:.3f}")
    print(f"  entropy Pearson r  trained {t['entropy_pearson_r']:.3f}   untrained {b['entropy_pearson_r']:.3f}")
    print(f"  entropy MAE (nats) trained {t['entropy_mae_nats']:.3f}   untrained {b['entropy_mae_nats']:.3f}")
    print(f"  KL(teacher||stud)  trained {t['kl_teacher_student_nats_per_byte']:.3f}   untrained {b['kl_teacher_student_nats_per_byte']:.3f}")
    print(f"  bits/byte  student {t['bits_per_byte_student']:.3f}  vs teacher {t['bits_per_byte_teacher']:.3f}")
    print(f"[eval] wrote {out/'distill_eval.json'}")


if __name__ == "__main__":
    main()
