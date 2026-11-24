"""Expand the pretraining corpus with MORE GiftEvalPretrain shards, streamed one shard at a time.

data/pretrain (v1-v3) holds every small source plus ONE shard of each large source, capped at 60k series per
source: ~1.2B points, well under 1% of the ~975 GB corpus. Telos-Small trained on GiftEvalPretrain too,
with about 20 times our training samples. v3 gets worse after 10k steps (it re-reads the same data), so we
are data-limited.

This script picks shards NOT used in data/pretrain and goes round-robin over the large sources (a different
source each time, for diversity). For each shard: download -> pack (<= SERIES_PER_SHARD series, each capped
at MAX_LEN points) -> delete the raw shard. It stops at --budget-points, or when free disk drops below
--min-free-gb. It writes data/pretrain2/{values.f32,index.npy,meta.json} in the same format, and RealStore
merges the roots by source name ("data/pretrain,data/pretrain2"). GiftEvalPretrain is built leakage-free
w.r.t. GIFT-Eval, so no test series enter.

  python prep_pretrain2.py --budget-points 1.5e9 --min-free-gb 7
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from prep_pretrain import FAMILY_KEEP, MAX_LEN, REPO, SMALL_MB, iter_series

SERIES_PER_SHARD = 25000
MAX_SHARD_GB = 1.5
SKIP_FAMILIES = {"cmip6", "era5", "largest"}  # near-duplicate climate/yearly families: little new diversity


def plan(seed: int) -> list:
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True) if hasattr(f, "size")]
    by_dir: dict[str, list] = defaultdict(list)
    for f in files:
        if "/" in f.path and f.path.split("/")[-1].startswith("data-") and f.path.endswith(".arrow"):
            by_dir[f.path.split("/")[0]].append(f)
    rng = np.random.default_rng(seed)
    queues = {}
    for d, fs in sorted(by_dir.items()):
        fam = re.sub(r"_\d{4}$", "", d)
        if fam in SKIP_FAMILIES or fam in FAMILY_KEEP:
            continue
        size = sum(f.size for f in fs)
        if size <= SMALL_MB * 1e6:  # already fully used
            continue
        used = min(fs, key=lambda f: f.size).path  # the shard data/pretrain took
        rest = [f for f in fs if f.path != used and f.size <= MAX_SHARD_GB * 1e9]
        if rest:
            queues[d] = [rest[i] for i in rng.permutation(len(rest))]
    order = []
    while any(queues.values()):  # round-robin over sources
        for d in sorted(queues):
            if queues[d]:
                order.append(queues[d].pop(0))
    return order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-points", type=float, default=1.5e9)
    ap.add_argument("--min-free-gb", type=float, default=7.0)
    ap.add_argument("--raw", default="data/pretrain_raw2")
    ap.add_argument("--out", default="data/pretrain2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download

    order = plan(a.seed)
    print(f"{len(order)} candidate shards from {len({f.path.split('/')[0] for f in order})} sources", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(a.raw)
    rng = np.random.default_rng(a.seed + 1)
    values = open(out / "values.f32", "wb")
    index, sources, stats, offset = [], [], {}, 0

    def save():
        values.flush()
        np.save(out / "index.npy", np.asarray(index, dtype=np.int64).reshape(-1, 3))
        json.dump({"sources": sources, "stats": stats, "total_points": offset}, open(out / "meta.json", "w"), indent=1)

    for f in order:
        free = shutil.disk_usage(".").free / 1e9
        if offset >= a.budget_points or free - f.size / 1e9 < a.min_free_gb:
            print(f"stop: points={offset:,d} free={free:.1f} GB next={f.size / 1e9:.2f} GB", flush=True)
            if offset >= a.budget_points or free < a.min_free_gb + 0.5:
                break
            continue  # this shard is too big for the remaining disk; try smaller ones
        src = f.path.split("/")[0]
        try:
            p = hf_hub_download(REPO, f.path, repo_type="dataset", local_dir=str(raw))
        except Exception as e:  # noqa: BLE001
            print(f"download failed {f.path}: {e}", flush=True)
            continue
        if src not in sources:
            sources.append(src)
            stats[src] = {"series": 0, "points": 0, "freq": None}
        sid = sources.index(src)
        n = 0
        try:
            for x, fq in iter_series(Path(p)):
                stats[src]["freq"] = stats[src]["freq"] or fq
                if n >= SERIES_PER_SHARD:
                    break
                ok = np.isfinite(x)
                if ok.sum() < 16:
                    continue
                xv = x[ok]
                if np.nanmax(xv) - np.nanmin(xv) <= 1e-8 * (1 + abs(xv[0])):
                    continue
                if len(x) > MAX_LEN:
                    s = int(rng.integers(0, len(x) - MAX_LEN + 1))
                    x = x[s : s + MAX_LEN]
                x.astype(np.float32).tofile(values)
                index.append((offset, len(x), sid))
                offset += len(x)
                n += 1
                stats[src]["series"] += 1
                stats[src]["points"] += len(x)
        except Exception as e:  # noqa: BLE001
            print(f"read failed {f.path}: {e}", flush=True)
        Path(p).unlink(missing_ok=True)
        save()
        print(f"{f.path:70s} +{n:6d} series  total={offset / 1e9:.3f}B pts  free={shutil.disk_usage('.').free / 1e9:.1f} GB",
              flush=True)
    save()
    values.close()
    shutil.rmtree(raw, ignore_errors=True)
    print(f"done: {len(index)} series, {offset:,d} points, {len(sources)} sources", flush=True)


if __name__ == "__main__":
    main()
