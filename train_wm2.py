"""Train Alma-WM2: warm-start the world model from Alma-WM (wm1); frozen forecasting path from Alma v2.

python train_wm2.py --out runs/alma_wm2 --init-from runs/alma_wm1/ema_8000.pt --base runs/alma_v2/ema_6000.pt
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

from data import PATCH
from data_wm import WMStream
from model import config_from_dict
from model_wm2 import AlmaWM2, WM2Config, wm2_config_dict, wm2_losses
from simulators import TRAIN_KINDS_V2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--target-ema", type=float, default=0.998)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--p-ctrl", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--real", default="data/pretrain")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    src = torch.load(args.init_from, map_location="cpu", weights_only=False)
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg, wcfg = config_from_dict(src["config"]["base"]), WM2Config()
    model = AlmaWM2(cfg, wcfg)
    sd = src["ema"]
    own = model.state_dict()
    P, nq = cfg.patch, cfg.n_q
    for k, v in sd.items():
        if k.startswith("dec."):
            t_ = own[k]
            if k in ("dec.fc1.weight", "dec.skip.weight") and t_.shape[1] != v.shape[1]:
                t_.zero_()
                t_[: v.shape[0], : v.shape[1]] = v  # prior inputs start with zero weight
                if k == "dec.skip.weight":
                    t_[P * nq :] = 0
            elif k in ("dec.fc2.weight", "dec.skip.weight", "dec.fc2.bias", "dec.skip.bias") and t_.shape[0] != v.shape[0]:
                t_[: v.shape[0]] = v
            else:
                t_.copy_(v)
        else:
            own[k].copy_(v)
    for k, v in base.get("ema", base.get("model")).items():
        own["fc_base." + k].copy_(v)
    model.load_state_dict(own)
    model.to(dev)
    target = copy.deepcopy(model.enc).eval()  # JEPA target encoder (EMA of the context encoder)
    ema = copy.deepcopy(model).eval()  # evaluation weights
    for p in list(target.parameters()) + list(ema.parameters()):
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    trn = model.trainable()
    decay = [p for p in trn if p.ndim >= 2]
    no_decay = [p for p in trn if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    json.dump({"args": vars(args), "config": wm2_config_dict(cfg, wcfg), "params": n_params,
               "trainable": sum(p.numel() for p in trn),
               "init_from": args.init_from}, open(out / "run.json", "w"), indent=1)
    print(f"Alma-WM2 params={n_params / 1e6:.2f}M trainable={sum(p.numel() for p in trn) / 1e6:.2f}M device={dev}", flush=True)

    def lr_at(s: int) -> float:
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        pr = (s - args.warmup) / max(1, args.steps - args.warmup)
        return 1e-2 * args.lr + (args.lr - 1e-2 * args.lr) * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))

    dl = DataLoader(WMStream(args.real, seed=args.seed + 101, window=args.window, p_ctrl=args.p_ctrl,
                             kinds=TRAIN_KINDS_V2),
                    batch_size=args.bs, num_workers=args.workers, persistent_workers=args.workers > 0,
                    prefetch_factor=4 if args.workers > 0 else None)
    hz, N = cfg.horizon, args.window // PATCH
    log = open(out / "log.jsonl", "a")
    step, t0, acc = 0, time.time(), {}
    for x, m, u, um in dl:
        if step >= args.steps:
            break
        x, m, u, um = (t.to(dev, non_blocking=True) for t in (x, m, u, um))
        xpad = torch.cat([x, x.new_zeros(x.shape[0], hz)], 1)
        mpad = torch.cat([m, m.new_zeros(m.shape[0], hz)], 1)
        y = xpad[:, PATCH:].unfold(1, hz, PATCH)[:, :N]
        ym = mpad[:, PATCH:].unfold(1, hz, PATCH)[:, :N]
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        L = wm2_losses(model, target, x, m, u, um, y, ym)
        opt.zero_grad(set_to_none=True)
        L["total"].backward()
        gn = torch.nn.utils.clip_grad_norm_(trn, 1.0)
        opt.step()
        with torch.no_grad():
            for pt, pm in zip(target.parameters(), model.enc.parameters()):
                pt.lerp_(pm, 1 - args.target_ema)
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        for k, v in L.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if step % 50 == 0:
            rec = {"step": step, **{k: round(v / 50, 5) for k, v in acc.items()}, "gnorm": round(float(gn), 4),
                   "lr": lr_at(step), "seq_s": round(50 * args.bs / (time.time() - t0), 2)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            acc, t0 = {}, time.time()
        if step % args.save_every == 0 or step == args.steps:
            strip = lambda sd: {k: v for k, v in sd.items() if not k.startswith("fc_base.")}  # frozen: from --base
            ck = {"ema": strip(ema.state_dict()), "config": wm2_config_dict(cfg, wcfg), "step": step,
                  "base_from": args.base}
            torch.save(ck, out / f"ema_{step}.pt")
            torch.save({**ck, "model": strip(model.state_dict()), "target": target.state_dict(),
                        "opt": opt.state_dict()}, out / "last.pt")
            old = sorted(out.glob("ema_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[:-3]
            for p_ in old:  # disk is tight: keep the 3 latest EMA snapshots
                p_.unlink()
            print(f"SAVED ema_{step}.pt", flush=True)


if __name__ == "__main__":
    main()
