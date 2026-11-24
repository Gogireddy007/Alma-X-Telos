"""Train Alma v3 (12 x 512, stride-32 coarse path, JEPA objective), grown exactly from Alma v2.

python train_v3.py --out runs/alma_v3 --init-from runs/alma_v2/ema_6000.pt --steps 20000
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import PATCH, TrainStream
from model import config_from_dict
from model_v3 import AlmaV3, JEPAHead, V3Config, balanced_quantile_loss, grow_from_v2, jepa_loss, v3_config_dict
from train import quick_eval


def right_align(x: torch.Tensor, m: torch.Tensor):
    """Shift each row so its last observed value sits at the end (short series fill the fine path)."""
    T = x.shape[1]
    ar = torch.arange(T, device=x.device)
    last = torch.where(m, ar, torch.full_like(ar, -1)).amax(1)
    shift = (T - 1 - last).clamp_min(0)
    idx = (ar[None] - shift[:, None]) % T
    return x.gather(1, idx), m.gather(1, idx) & (ar[None] >= shift[:, None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--micro-bs", type=int, default=8, help="gradient-accumulation chunk (fits 16 GB unified memory)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--target-ema", type=float, default=0.998)
    ap.add_argument("--p-long", type=float, default=0.35, help="fraction of batches with 8k context (coarse path)")
    ap.add_argument("--gamma", type=float, default=0.5, help="long-horizon weight exponent")
    ap.add_argument("--jepa-w", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--real", default="data/pretrain")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-ctx", type=int, default=8192)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-v3", default=None, help="fine-tune: load a v3 EMA checkpoint (weights + JEPA head)")
    ap.add_argument("--sampler", default="sqrt", choices=["sqrt", "bucket"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--loss-norm", default="token", choices=["token", "series", "mix"])
    ap.add_argument("--p-short", type=float, default=0.0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device

    cfg = V3Config()
    model = AlmaV3(cfg)
    src = torch.load(args.init_from, map_location="cpu", weights_only=False)
    idx = grow_from_v2(model, src.get("ema", src.get("model")), config_from_dict(src["config"]))
    jepa = JEPAHead(cfg)
    if args.init_v3:
        v = torch.load(args.init_v3, map_location="cpu", weights_only=False)
        model.load_state_dict(v["ema"])
        if "jepa" in v:
            jepa.load_state_dict(v["jepa"])
        print(f"fine-tune from {args.init_v3} (step {v.get('step')})", flush=True)
    model.to(dev)
    jepa.to(dev)
    target = copy.deepcopy(model).eval()
    ema = copy.deepcopy(model).eval()
    for p in list(target.parameters()) + list(ema.parameters()):
        p.requires_grad_(False)
    params = list(model.parameters()) + list(jepa.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in params if p.ndim >= 2], "weight_decay": args.wd},
                             {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    step = 0
    if args.resume and (out / "last.pt").exists():
        r = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"])
        jepa.load_state_dict(r["jepa"])
        target.load_state_dict(r["target"])
        ema.load_state_dict(r["ema"])
        opt.load_state_dict(r["opt"])
        step = int(r["step"])
        print(f"RESUMED at step {step}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    json.dump({"args": vars(args), "config": v3_config_dict(cfg), "params": n_params,
               "jepa_params": sum(p.numel() for p in jepa.parameters()), "grow_idx": idx}, open(out / "run.json", "w"),
              indent=1)
    print(f"Alma v3 params={n_params / 1e6:.2f}M (+JEPA head {sum(p.numel() for p in jepa.parameters()) / 1e6:.2f}M) "
          f"device={dev} start={step}", flush=True)

    def lr_at(s: int) -> float:
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        pr = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 1e-2 * args.lr + (args.lr - 1e-2 * args.lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))

    W = 2 * cfg.fine_len
    dl = DataLoader(TrainStream(args.real, seed=step + 23 + 1000 * args.seed, window=W, sampler=args.sampler,
                                alpha=args.alpha, p_short=args.p_short), batch_size=args.bs, num_workers=args.workers,
                    persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    hz, nf = cfg.horizon, cfg.fine_len // PATCH
    gen = torch.Generator().manual_seed(1000 + step)
    log = open(out / "log.jsonl", "a")
    t0, acc = time.time(), {}
    for x, m in dl:
        if step >= args.steps:
            break
        x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        long = bool(torch.rand(1, generator=gen) < args.p_long)
        if long:
            x, m = right_align(x, m)
        else:
            x, m = x[:, : cfg.fine_len], m[:, : cfg.fine_len]
        T = x.shape[1]
        xpad = torch.cat([x, x.new_zeros(x.shape[0], hz)], 1)
        mpad = torch.cat([m, m.new_zeros(m.shape[0], hz)], 1)
        y = xpad[:, PATCH:].unfold(1, hz, PATCH)[:, : T // PATCH][:, -nf:]
        ym = mpad[:, PATCH:].unfold(1, hz, PATCH)[:, : T // PATCH][:, -nf:]
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        nb = x.shape[0]
        loss = l_fc = l_j = l_v = 0.0
        for c0 in range(0, nb, args.micro_bs):
            xs, ms, ys, yms = (a[c0 : c0 + args.micro_bs] for a in (x, m, y, ym))
            fr = xs.shape[0] / nb
            h, mu, sd = model.encode(xs, ms)
            q = model.head(h).view(h.shape[0], h.shape[1], hz, cfg.n_q)
            c_fc = balanced_quantile_loss(q, mu, sd, ys, yms, cfg.quantiles, cfg.asinh_clip, args.gamma, args.loss_norm)
            with torch.no_grad():
                zt, _, _ = target.encode(xs, ms)
            tok = ms[:, -cfg.fine_len :].view(ms.shape[0], nf, PATCH).any(-1)
            tok = tok & tok.cummax(-1).values
            c_j, c_v = jepa_loss(jepa, h, zt, tok, cfg.jepa_K)
            c_loss = c_fc + args.jepa_w * (c_j + c_v)
            (c_loss * fr).backward()
            loss, l_fc, l_j, l_v = (a + fr * b.detach() for a, b in ((loss, c_loss), (l_fc, c_fc), (l_j, c_j), (l_v, c_v)))
            del h, mu, sd, q, zt, c_loss
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            for pt, pm in zip(target.parameters(), model.parameters()):
                pt.lerp_(pm, 1 - args.target_ema)
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        for k, v in (("loss", loss), ("fc", l_fc), ("jepa", l_j), ("var", l_v)):
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        acc["long"] = acc.get("long", 0.0) + float(long)
        if step % 50 == 0:
            rec = {"step": step, **{k: round(v / 50, 5) for k, v in acc.items()}, "gnorm": round(float(gn), 4),
                   "lr": lr_at(step), "seq_s": round(50 * args.bs / (time.time() - t0), 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            acc, t0 = {}, time.time()
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "jepa": jepa.state_dict(), "target": target.state_dict(),
                        "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step, "config": v3_config_dict(cfg)},
                       out / "last.pt")
        if step % args.eval_every == 0 or step == args.steps:
            r = {"step": step, **quick_eval(ema, dev, context_len=args.eval_ctx)}
            ema.eval()
            print("EVAL", json.dumps(r), flush=True)
            log.write(json.dumps({"eval": r}) + "\n")
            log.flush()
            torch.save({"ema": ema.state_dict(), "jepa": jepa.state_dict(), "config": v3_config_dict(cfg),
                        "step": step}, out / f"ema_{step}.pt")
            t0 = time.time()


if __name__ == "__main__":
    main()
