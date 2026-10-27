"""Online knowledge distillation: train the from-scratch StudentBLT on the frozen BLT
teacher's next-byte distribution.

For each minibatch of <=512-byte windows we run the teacher once (no grad) to get soft
targets and train the student to match them:

    loss = alpha * CE(student, true_next_byte)
         + (1 - alpha) * T^2 * KL( softmax(teacher/T) || softmax(student/T) )

evaluated only at positions that predict a real byte. The teacher distribution is the
supervision; the hard-label CE term is a small anchor (alpha defaults low). "Online"
means the teacher is evaluated on the fly, so no multi-terabyte dump of 260-way
distributions is ever written to disk.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from escape_distill.student_blt import StudentBLT, StudentConfig
from escape_distill.data import load_samples, build_windows, WindowLoader
from escape_distill.teacher import load_teacher, teacher_logits


def kd_loss(student_logits, teacher_logits_, input_ids, predict_mask, temperature, alpha):
    """Masked distillation loss. Returns (loss, soft_kl, hard_ce) scalars."""
    B, L, V = student_logits.shape
    # position j predicts token j+1 -> compare logits[:, :-1] with next ids [:, 1:]
    s = student_logits[:, :-1].reshape(-1, V)
    t = teacher_logits_[:, :-1].reshape(-1, V)
    tgt = input_ids[:, 1:].reshape(-1)
    m = predict_mask[:, :-1].reshape(-1)
    s, t, tgt = s[m], t[m], tgt[m]

    T = temperature
    log_p_student = F.log_softmax(s / T, dim=-1)
    p_teacher = F.softmax(t / T, dim=-1)
    soft_kl = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)
    hard_ce = F.cross_entropy(s, tgt)
    loss = alpha * hard_ce + (1.0 - alpha) * soft_kl
    return loss, soft_kl.detach(), hard_ce.detach()


def main():
    ap = argparse.ArgumentParser(description="Distill the BLT teacher into StudentBLT.")
    ap.add_argument("--ckpt-dir", required=True, help="teacher checkpoint dir (has patcher.safetensors, config.json)")
    ap.add_argument("--data", required=True, help="manifest_ext.parquet path, a dir of *.bin, or 'golden'")
    ap.add_argument("--out", required=True, help="output directory for the student + logs")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--domains", default=None, help="comma-separated subset e.g. py,prose,cpp")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.1, help="weight of hard-label CE (rest is soft KD)")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--init-from", default=None,
                    help="checkpoint dir (patcher.safetensors + config.json): initialise the student "
                         "from the teacher's weights (domain-adaptation arm). Use --dim 768 --layers 14 "
                         "--heads 12 for the teacher architecture; tie_embeddings is disabled automatically.")
    ap.add_argument("--teacher-dtype", default="float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    domains = tuple(args.domains.split(",")) if args.domains else None
    samples = load_samples(args.data, split=args.train_split, limit=args.limit_files,
                           domains=domains, seed=args.seed)
    windows = build_windows(samples)
    if not windows:
        raise SystemExit("no training windows found")
    loader = WindowLoader(windows, args.batch_size, device, shuffle=True, seed=args.seed)

    cfg = StudentConfig(dim=args.dim, n_layers=args.layers, n_heads=args.heads)
    if args.init_from:
        cfg.tie_embeddings = False
    student = StudentBLT(cfg).to(device)
    if args.init_from:
        from escape_distill.teacher import _load_hf_state
        state = _load_hf_state(Path(args.init_from), str(device), "float32")
        top = {"embed_tokens.weight": "tok_emb.weight", "norm.weight": "final_norm.weight",
               "lm_head.weight": "lm_head.weight"}
        per_layer = {"self_attn.q_proj.weight": "attn.wq.weight", "self_attn.k_proj.weight": "attn.wk.weight",
                     "self_attn.v_proj.weight": "attn.wv.weight", "self_attn.o_proj.weight": "attn.wo.weight",
                     "mlp.gate_proj.weight": "ffn.gate.weight", "mlp.up_proj.weight": "ffn.up.weight",
                     "mlp.down_proj.weight": "ffn.down.weight",
                     "input_layernorm.weight": "attn_norm.weight",
                     "post_attention_layernorm.weight": "ffn_norm.weight"}
        sd = {}
        for k, v in state.items():
            if k in top:
                sd[top[k]] = v
            elif k.startswith("layers."):
                _, idx, rest = k.split(".", 2)
                sd[f"blocks.{idx}.{per_layer[rest]}"] = v
        missing, unexpected = student.load_state_dict(sd, strict=False)
        if missing:
            raise SystemExit(f"--init-from: {len(missing)} missing keys, first {missing[:6]}")
        print(f"[distill] initialised from teacher checkpoint ({len(sd)} keys; {len(unexpected)} unused)")
    teacher = load_teacher(args.ckpt_dir, device=str(device), dtype=args.teacher_dtype)

    total_steps = (args.max_steps if args.max_steps else len(loader) * args.epochs)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        import math
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    print(f"[distill] files={len(samples)} windows={len(windows)} steps={total_steps} "
          f"student_params={student.num_parameters():,} device={device}")

    log = []
    step = 0
    t0 = time.time()
    student.train()
    done = False
    for epoch in range(args.epochs):
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            tlogits = teacher_logits(teacher, batch["input_ids"])
            slogits = student(batch["input_ids"])
            loss, soft_kl, hard_ce = kd_loss(slogits, tlogits, batch["input_ids"],
                                             batch["predict_mask"], args.temperature, args.alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            if step % args.log_every == 0 or step == total_steps - 1:
                rec = {"step": step, "epoch": epoch, "loss": float(loss),
                       "soft_kl": float(soft_kl), "hard_ce_nats": float(hard_ce),
                       "lr": lr_at(step), "elapsed_s": round(time.time() - t0, 1)}
                log.append(rec)
                print(f"  step {step:5d}/{total_steps}  loss {rec['loss']:.4f}  "
                      f"KL {rec['soft_kl']:.4f}  CE {rec['hard_ce_nats']:.3f} nats  lr {rec['lr']:.2e}")
            step += 1
            if args.max_steps and step >= args.max_steps:
                done = True
                break
        if done:
            break

    ckpt = {"model_state": student.state_dict(), "config": asdict(cfg),
            "train_args": vars(args), "steps": step}
    torch.save(ckpt, out / "student.pt")
    (out / "train_log.json").write_text(json.dumps(
        {"args": vars(args), "student_params": student.num_parameters(),
         "n_files": len(samples), "n_windows": len(windows), "steps": step, "log": log}, indent=2))
    print(f"[distill] saved student to {out/'student.pt'}  ({step} steps, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
