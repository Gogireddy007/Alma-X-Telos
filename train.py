"""Train Alma.  python train.py --out runs/alma_s --steps 24000"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # data workers each use one BLAS thread (oversubscription was 300x slower)

import argparse
import copy
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import PATCH, WINDOW, TrainStream
from model import Alma, AlmaConfig, config_to_dict, quantile_loss


def quick_eval(model: Alma, device: str, cache: str = "data/gift_cache", n_win: int = 64) -> dict:
    """Fast proxy on cached GIFT-Eval windows (first n_win per task), relative to Seasonal Naive."""
    from eval_gift import QUANTILES, score, seasonal_naive
    from forecast import AlmaForecaster

    f = AlmaForecaster(model, device, symmetric=False, batch_size=128)
    rels = []
    for p in sorted(glob.glob(f"{cache}/*__1000.npz")):
        z = np.load(p, allow_pickle=True)
        ctx, lab, H, m = list(z["contexts"])[:n_win], z["labels"][:n_win], int(z["H"]), int(z["season"])
        sn = seasonal_naive(ctx, H, m)
        s_sn = score(ctx, lab, np.repeat(sn[..., None], 9, -1), m)["crps"]
        q = np.sort(f.predict(ctx, H, QUANTILES), -1)
        rels.append(score(ctx, lab, q, m)["crps"] / max(s_sn, 1e-12))
    model.train()
    r = np.array(rels)
    r = r[np.isfinite(r) & (r > 0)]
    return {"quick_tasks": int(len(r)), "quick_rel_crps": float(np.exp(np.log(r).mean())) if len(r) else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--window", type=int, default=WINDOW, help="training window length (steps)")
    ap.add_argument("--horizon", type=int, default=512, help="direct forecast horizon per token")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--real", default="data/pretrain")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="warm-start weights (EMA) from a checkpoint; horizon may grow")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    cfg = AlmaConfig(patch=PATCH, horizon=args.horizon, d_model=args.d, n_layers=args.layers, n_heads=args.d // 64)
    model = Alma(cfg).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    step = 0
    if args.init_from and not args.resume:
        src = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = src.get("ema", src.get("model"))
        h_old, nq = src["config"]["horizon"], cfg.n_q
        own = model.state_dict()
        L_old, L_new = src["config"]["n_layers"], cfg.n_layers
        if L_new > L_old:  # depth growth: keep old blocks in order, insert identity blocks (zero output proj)
            per = L_new / L_old
            new_idx = [int(round(j * per)) for j in range(L_old)]
            remap = {}
            for k, v in sd.items():
                if k.startswith("blocks."):
                    j = int(k.split(".")[1])
                    remap[f"blocks.{new_idx[j]}." + k.split(".", 2)[2]] = v
                else:
                    remap[k] = v
            sd = remap
            for i in set(range(L_new)) - set(new_idx):
                own[f"blocks.{i}.o.weight"].zero_()
                own[f"blocks.{i}.w3.weight"].zero_()
            print(f"depth growth {L_old} -> {L_new}: old blocks at {new_idx}", flush=True)
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
            elif k.startswith("head.") and v.shape[0] == h_old * nq:  # grow horizon: tile old step weights
                rep = -(-cfg.horizon // h_old)
                grown = v.view(h_old, nq, *v.shape[1:]).repeat(rep, *([1] * v.dim()))[: cfg.horizon]
                own[k] = grown.reshape(cfg.horizon * nq, *v.shape[1:])
            else:
                print("init-from: skipped", k, tuple(v.shape), "->", tuple(own[k].shape) if k in own else None)
        model.load_state_dict(own)
        ema.load_state_dict(own)
        print(f"warm-started from {args.init_from} (horizon {h_old} -> {cfg.horizon})", flush=True)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    json.dump({"args": vars(args), "config": config_to_dict(cfg), "params": n_params}, open(out / "run.json", "w"), indent=1)
    print(f"params={n_params / 1e6:.2f}M device={dev} start_step={step}", flush=True)

    def lr_at(s: int) -> float:
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        pr = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 1e-2 * args.lr + (args.lr - 1e-2 * args.lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))

    dl = DataLoader(TrainStream(args.real, seed=step + 17, window=args.window), batch_size=args.bs, num_workers=args.workers,
                    persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)
    hz, N = cfg.horizon, args.window // PATCH
    log = open(out / "log.jsonl", "a")
    t0, run_loss = time.time(), 0.0
    for x, m in dl:
        if step >= args.steps:
            break
        x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        xpad = torch.cat([x, x.new_zeros(x.shape[0], hz)], 1)
        mpad = torch.cat([m, m.new_zeros(m.shape[0], hz)], 1)
        y = xpad[:, PATCH:].unfold(1, hz, PATCH)[:, :N]
        ym = mpad[:, PATCH:].unfold(1, hz, PATCH)[:, :N]
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        q, mu, sd = model(x, m)
        loss = quantile_loss(q, mu, sd, y, ym, cfg.quantiles, cfg.asinh_clip)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        run_loss += loss.item()
        if step % 50 == 0:
            rec = {"step": step, "loss": run_loss / 50, "gnorm": float(gn), "lr": lr_at(step),
                   "seq_s": 50 * args.bs / (time.time() - t0), "elapsed_h": None}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run_loss, t0 = 0.0, time.time()
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        "config": config_to_dict(cfg)}, out / "last.pt")
        if step % args.eval_every == 0 or step == args.steps:
            r = {"step": step, **quick_eval(ema, dev)}
            print("EVAL", json.dumps(r), flush=True)
            log.write(json.dumps({"eval": r}) + "\n")
            log.flush()
            torch.save({"ema": ema.state_dict(), "config": config_to_dict(cfg), "step": step}, out / f"ema_{step}.pt")
            t0 = time.time()


if __name__ == "__main__":
    main()
