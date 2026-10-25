"""Golden-fixture C-v1-style evaluation: distilled student vs frozen BLT teacher.

Read-only with respect to the ESCAPE repository (imports its modules; writes only
results under the experiment dir passed as OUT). Streams scored per domain:

  teacher_native   BLT patcher native regime: [BOS]+bytes, swa512 banded mask, float32.
                   Files are << 8192 tokens -> no chunking needed.
  teacher_512      SAME weights with the student's context budget: per-byte fresh
                   windows of <=512 preceding bytes (stride-1). For files <=512 bytes
                   this equals teacher_native exactly (same input per byte, causal ==
                   swa512 for <=512-token sequences, fidelity gate G3), so it is copied,
                   not recomputed. CONTROL for context-length effects only.
  student          distilled StudentBLT (14.26M params, commit c4384eb), per-byte fresh
                   windows of <=512 bytes (its trained regime, absolute RoPE 0..511).
  baselines        newline / indent_change / word_start (structure-blind, deterministic)
  null             fixed-EOF density-matched permutation null per file per model stream,
                   R=10,000 sequential draws per file, pooled counts per permutation ->
                   exact pooled permutation p-value.

Scoring: interior entropy-triggered starts; one-to-one matching at the frozen k=0
(results/final/c-v1-2721954 calibration record); micro-pooled P/R/F1; paired file
percentile bootstrap (B=2000). N=6 files per domain -> CIs are wide; this is a
golden-fixture variant experiment, NOT the held-out test split.
"""
import argparse
import hashlib
import json
import shutil
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
from escape_scoring.contract import WRAPPERS  # noqa: E402
from escape_scoring.metrics import match, metrics, rng_for, shuffled_boundaries  # noqa: E402
from stream_a.patching import bytes_to_token_ids, compute_patches  # noqa: E402
from stream_a.reference_lm import ReferenceEntropyModel, to_reference_names  # noqa: E402

TAU = 1.335442066192627
SEED = 20260927
N_PERMS = 10000
N_BOOT = 2000
MAX_CTX = 512
BATCH_WINDOWS = 4
CKPT_DIR = REPO / "checkpoints/itazap__blt-1b-hf/91aa6b8e168046ad91e517d2191e1e974f50cc01"
GOLDEN = REPO / "golden_fixtures"


def load_teacher():
    from safetensors.torch import load_file
    cfg = json.loads((CKPT_DIR / "config.json").read_text())
    pc = cfg.get("patcher_config", cfg)
    raw = load_file(str(CKPT_DIR / "patcher.safetensors"), device="cpu")
    state = {k[len("model.patcher."):]: v.to(torch.float32) for k, v in raw.items()}
    ref = ReferenceEntropyModel(
        to_reference_names(state), n_layers=pc["num_hidden_layers"], dim=pc["hidden_size"],
        n_heads=pc["num_attention_heads"], norm_eps=pc.get("norm_eps", 1e-5),
        rope_theta=pc.get("rope_theta", 10000.0),
        max_seqlen=pc.get("max_position_embeddings", 8192), device="cpu", dtype=torch.float32)
    return ref, pc


def banded_mask(L):
    idx = torch.arange(L)
    diff = idx[:, None] - idx[None, :]
    return ((diff >= 0) & (diff < 512))[None, None]


@torch.inference_mode()
def teacher_native_entropy(model, content: bytes) -> np.ndarray:
    ids = torch.from_numpy(bytes_to_token_ids(content)).unsqueeze(0)
    logits = model.logits(ids, mask=banded_mask(ids.shape[1])).float()
    logp = torch.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)[0].numpy()
    return ent[: len(content)].astype(np.float32)


def per_byte_entropy(model_fn, content: bytes):
    """Stride-1 fresh windows; returns entropy float32[n], nll_nats, wall_s."""
    n = len(content)
    ent = np.zeros(n, dtype=np.float32)
    nll = 0.0
    t0 = time.time()
    for s in range(0, n, BATCH_WINDOWS):
        chunk = list(range(s, min(s + BATCH_WINDOWS, n)))
        L = max(min(i, MAX_CTX) for i in chunk) + 1
        batch = torch.full((len(chunk), L), 0, dtype=torch.long)
        for r, i in enumerate(chunk):
            ctx = content[max(0, i - MAX_CTX): i]
            batch[r, 0] = 1
            batch[r, 1: len(ctx) + 1] = torch.from_numpy(np.frombuffer(ctx, dtype=np.uint8).astype(np.int64) + 4)
        logits = model_fn(batch).float()
        for r, i in enumerate(chunk):
            pos = min(i, MAX_CTX)
            logp = torch.log_softmax(logits[r, pos], -1)
            ent[i] = -float((logp.exp() * logp).sum())
            nll += -float(logp[int(content[i]) + 4])
    return ent, nll, time.time() - t0


def interior_starts(entropies: np.ndarray, tau: float = TAU) -> list:
    patches = compute_patches(entropies, tau)
    return sorted({int(s) for s, t in zip(patches.starts, patches.triggers)
                   if t == "entropy" and 0 < int(s) < len(entropies)})


def c_v1_patches(entropies: np.ndarray, tau: float = TAU) -> list:
    p = compute_patches(entropies, tau)
    starts, triggers = [int(s) for s in p.starts], [str(t) for t in p.triggers]
    n = len(entropies)
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else n
        reason = {"entropy": "entropy", "length_cap": "max_length"}[triggers[i + 1]] if i + 1 < len(starts) else "eof"
        out.append({"start_byte": s, "end_byte": e, "termination_reason": reason})
    return out


def targets_from_parquet(path: Path, n_bytes: int):
    import pandas as pd
    df = pd.read_parquet(path)
    df = df[(df["depth"] > 0) & (~df["node_type"].str.lower().isin(WRAPPERS))
            & (df["end_byte"] > df["start_byte"])
            & (df["start_byte"] > 0) & (df["start_byte"] < n_bytes)]
    df = df.sort_values(["depth", "end_byte", "node_type"], ascending=[True, False, True])
    seen, out = set(), []
    for s in df["start_byte"]:
        s = int(s)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/student_cv1_golden_v1"))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    teacher, pc = load_teacher()
    print(f"[setup] teacher: {pc['num_hidden_layers']}x{pc['hidden_size']} "
          f"{pc['num_attention_heads']} heads (reference backend, float32 cpu)", flush=True)

    ckpt = torch.load(HERE / "student.pt", map_location="cpu", weights_only=False)
    student = StudentBLT(StudentConfig(**ckpt["config"]))
    student.load_state_dict(ckpt["model_state"])
    student.eval()
    print(f"[setup] student: {student.num_parameters():,} params, trained {ckpt['steps']} steps", flush=True)

    domains = ("py", "cpp", "prose")
    files = {d: sorted((GOLDEN / d).glob("*.bin")) for d in domains}

    config = {
        "experiment": "student_cv1_golden_v1",
        "teacher": {"checkpoint": "itazap/blt-1b-hf",
                    "revision": "91aa6b8e168046ad91e517d2191e1e974f50cc01",
                    "backend": "stream_a.reference_lm (G3-verified re-implementation), float32 cpu",
                    "patcher_weights_sha256": hashlib.sha256((CKPT_DIR / "patcher.safetensors").read_bytes()).hexdigest()},
        "student": {"params": student.num_parameters(), "steps": ckpt["steps"],
                    "config": ckpt["config"],
                    "student_pt_sha256": hashlib.sha256((HERE / "student.pt").read_bytes()).hexdigest()},
        "context_policy": {
            "teacher_native": "swa512 banded mask over [BOS]+full file (native regime; no chunking needed, files << 8192 tokens)",
            "teacher_512": "stride-1 fresh windows, <=512 preceding bytes per prediction (student's budget; CONTROL only; equals native for files <=512B)",
            "student": "stride-1 fresh windows, <=512 preceding bytes, absolute RoPE 0..511 (its trained regime); no artificial 512-chunk resets"},
        "tau": TAU, "k": 0, "k_source": "frozen calibration results/final/c-v1-2721954/calibration.json (k=0, all languages)",
        "seed": SEED, "n_permutations": N_PERMS, "n_bootstrap": N_BOOT,
        "torch": torch.__version__, "device": "cpu",
    }

    model_streams = ("teacher_native", "teacher_512", "student")
    out = {"config": config, "domains": {}}

    for d in domains:
        fid_order = [p.stem for p in files[d]]
        ent = {s: {} for s in model_streams}
        starts = {s: {} for s in model_streams}
        patches = {s: {} for s in model_streams}
        bpb = {s: {} for s in model_streams}
        rt = {s: {} for s in model_streams}
        tgt, base = {}, {b: {} for b in ("newline", "indent_change", "word_start")}

        for p in files[d]:
            fid = p.stem
            content = p.read_bytes()
            n = len(content)
            tgt[fid] = targets_from_parquet(GOLDEN / d / f"{fid}.structure.parquet", n)
            base["newline"][fid] = sorted({o for o in newline(content) if 0 < o < n})
            base["indent_change"][fid] = sorted({o for o in indent_change(content) if 0 < o < n})
            base["word_start"][fid] = sorted({int(o) for o in word_start(content) if 0 < o < n})

            t = time.time()
            ent["teacher_native"][fid] = teacher_native_entropy(teacher, content)
            rt["teacher_native"][fid] = time.time() - t
            starts["teacher_native"][fid] = interior_starts(ent["teacher_native"][fid])
            patches["teacher_native"][fid] = c_v1_patches(ent["teacher_native"][fid])
            bpb["teacher_native"][fid] = None

            if n <= MAX_CTX:
                ent["teacher_512"][fid] = ent["teacher_native"][fid]
                starts["teacher_512"][fid] = starts["teacher_native"][fid]
                patches["teacher_512"][fid] = patches["teacher_native"][fid]
                bpb["teacher_512"][fid] = None
                rt["teacher_512"][fid] = 0.0
            else:
                e512, nll512, dt512 = per_byte_entropy(lambda b: teacher.logits(b, mask=None).float(), content)
                ent["teacher_512"][fid] = e512
                starts["teacher_512"][fid] = interior_starts(e512)
                patches["teacher_512"][fid] = c_v1_patches(e512)
                bpb["teacher_512"][fid] = nll512 / np.log(2) / n
                rt["teacher_512"][fid] = dt512

            eS, nllS, dtS = per_byte_entropy(lambda b: student(b), content)
            ent["student"][fid] = eS
            starts["student"][fid] = interior_starts(eS)
            patches["student"][fid] = c_v1_patches(eS)
            bpb["student"][fid] = nllS / np.log(2) / n
            rt["student"][fid] = dtS
            print(f"[run] {d}/{fid}: {n}B native={rt['teacher_native'][fid]:.2f}s "
                  f"t512={rt['teacher_512'][fid]:.2f}s student={dtS:.2f}s", flush=True)

        # ---- scoring ----
        rows = {}
        for name in model_streams + tuple(f"baseline_{b}" for b in ("newline", "indent_change", "word_start")):
            per = []
            for fid in fid_order:
                pred = starts[name][fid] if name in model_streams else base[name[len("baseline_"):]][fid]
                tp = len(match(pred, tgt[fid], 0))
                per.append([tp, len(pred) - tp, len(tgt[fid]) - tp])
            arr = np.asarray(per, dtype=float)
            m = metrics(arr.sum(axis=0))
            rows[name] = {"tp": int(arr[:, 0].sum()), "fp": int(arr[:, 1].sum()), "fn": int(arr[:, 2].sum()),
                          "precision": float(m["precision"]), "recall": float(m["recall"]), "f1": float(m["f1"]),
                          "n_files": len(fid_order), "per_file": per}

        # Pooled null per model stream, streaming over permutations (no per-rep storage).
        # teacher_512 reuses teacher_native's draw for a file whenever the patch layout
        # is identical (all files <=512B): same purpose string -> identical draws.
        null_f1 = {s: np.empty(N_PERMS) for s in model_streams}
        rngs = {(s, fid): rng_for(SEED, fid, f"student_cv1_{s}") for s in model_streams
                for fid in fid_order}
        for r in range(N_PERMS):
            counts = {s: [0, 0, 0] for s in model_streams}
            draws = {}
            for fid in fid_order:
                n = ent["teacher_native"][fid].shape[0]
                draws[(fid, "teacher_native")] = shuffled_boundaries(
                    patches["teacher_native"][fid], n, rngs[("teacher_native", fid)], eof_policy="fixed_eof")
                if patches["teacher_512"][fid] == patches["teacher_native"][fid]:
                    draws[(fid, "teacher_512")] = draws[(fid, "teacher_native")]
                else:
                    draws[(fid, "teacher_512")] = shuffled_boundaries(
                        patches["teacher_512"][fid], n, rngs[("teacher_512", fid)], eof_policy="fixed_eof")
                draws[(fid, "student")] = shuffled_boundaries(
                    patches["student"][fid], n, rngs[("student", fid)], eof_policy="fixed_eof")
                for s in model_streams:
                    t = len(match(draws[(fid, s)], tgt[fid], 0))
                    counts[s][0] += t
                    counts[s][1] += len(draws[(fid, s)]) - t
                    counts[s][2] += len(tgt[fid]) - t
            for s in model_streams:
                tp, fp, fn = counts[s]
                null_f1[s][r] = 2 * tp / max(1e-12, 2 * tp + fp + fn)
        for name in model_streams:
            pooled_obs = np.asarray(rows[name]["per_file"]).sum(axis=0)
            obs_f1 = float(metrics(pooled_obs)["f1"])
            rows[name]["f1_h0_mean"] = float(null_f1[name].mean())
            rows[name]["f1_margin"] = float(obs_f1 - null_f1[name].mean())
            rows[name]["f1_permutation_p"] = float((1 + np.sum(null_f1[name] >= obs_f1)) / (N_PERMS + 1))
            print(f"[null] {d}/{name}: h0_f1={rows[name]['f1_h0_mean']:.4f} p={rows[name]['f1_permutation_p']:.4g}", flush=True)

        rng_b = rng_for(SEED, f"{d}\0student_cv1", "file_bootstrap")
        for name in rows:
            arr = np.asarray(rows[name]["per_file"], dtype=float)
            draws = []
            for _ in range(N_BOOT):
                idx = rng_b.integers(0, len(arr), len(arr))
                draws.append(float(metrics(arr[idx].sum(axis=0))["f1"]))
            rows[name]["f1_ci_low"], rows[name]["f1_ci_high"] = (
                float(np.quantile(draws, .025)), float(np.quantile(draws, .975)))
        best_name = max(("newline", "indent_change", "word_start"),
                        key=lambda b: rows[f"baseline_{b}"]["f1"])
        for name in model_streams:
            a = np.asarray(rows[name]["per_file"], dtype=float)
            b = np.asarray(rows[f"baseline_{best_name}"]["per_file"], dtype=float)
            draws = []
            for _ in range(N_BOOT):
                idx = rng_b.integers(0, len(a), len(a))
                draws.append(float(metrics(a[idx].sum(axis=0))["f1"] - metrics(b[idx].sum(axis=0))["f1"]))
            rows[name]["strongest_baseline"] = best_name
            rows[name]["f1_minus_best_baseline"] = float(metrics(a.sum(axis=0))["f1"] - metrics(b.sum(axis=0))["f1"])
            rows[name]["f1_minus_best_baseline_ci"] = [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]

        diag = {}
        total_bytes = sum(len(p.read_bytes()) for p in files[d])
        for name in model_streams:
            all_ent = np.concatenate([ent[name][f] for f in fid_order])
            bb = [bpb[name][f] for f in fid_order if bpb[name][f] is not None]
            diag[name] = {
                "entropy_mean": float(all_ent.mean()), "entropy_median": float(np.median(all_ent)),
                "entropy_max": float(all_ent.max()), "frac_above_tau": float((all_ent.astype(np.float64) > TAU).mean()),
                "bytes_within_1e-3_of_tau": int((np.abs(all_ent.astype(np.float64) - TAU) < 1e-3).sum()),
                "interior_entropy_boundaries": int(sum(len(starts[name][f]) for f in fid_order)),
                "boundary_density": float(sum(len(starts[name][f]) for f in fid_order) / total_bytes),
                "bits_per_byte": float(np.mean(bb)) if bb else None,
                "runtime_s": float(sum(rt[name][f] for f in fid_order))}
        out["domains"][d] = {"fid_order": fid_order, "streams": rows, "diagnostics": diag,
                             "n_targets": int(rows["teacher_native"]["tp"] + rows["teacher_native"]["fn"])}

    (OUT / "scores.json").write_text(json.dumps(out, indent=2))
    (OUT / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copyfile(__file__, OUT / "run_golden_eval.py")
    print(f"\n[wrote] {OUT / 'scores.json'}")
    for d in domains:
        dres = out["domains"][d]
        print(f"\n=== {d}  (files={len(files[d])}, targets={dres['n_targets']}) ===")
        for name, r in dres["streams"].items():
            extra = ""
            if name in model_streams:
                extra = (f" | margin_vs_null={r['f1_margin']:+.4f} p={r['f1_permutation_p']:.4g}"
                         f" | vs_{r['strongest_baseline']}={r['f1_minus_best_baseline']:+.4f}"
                         f" [{r['f1_minus_best_baseline_ci'][0]:+.4f},{r['f1_minus_best_baseline_ci'][1]:+.4f}]")
            print(f"  {name:16s} F1={r['f1']:.4f} [{r['f1_ci_low']:.4f},{r['f1_ci_high']:.4f}] "
                  f"P={r['precision']:.4f} R={r['recall']:.4f} tp={r['tp']}{extra}")


if __name__ == "__main__":
    main()
