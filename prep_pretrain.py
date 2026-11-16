"""Download a diverse, disk-bounded slice of Salesforce/GiftEvalPretrain and pack it into a flat float32 memmap.

GiftEvalPretrain is ~975 GB and is built to be leakage-free w.r.t. the GIFT-Eval test set. We keep:
  * every source whose size is <= SMALL_MB (all shards),
  * one shard of every larger source,
  * only a few members of the near-duplicate yearly families (cmip6_*, era5_*, largest_*).
Then each series (multivariate targets are split into channels) is capped at MAX_LEN points (random window),
NaN-only / constant series are dropped, and everything goes into data/pretrain/{values.f32, index.npy, meta.json}.

Usage: python prep_pretrain.py [--budget-gb 7] [--keep-raw]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = "Salesforce/GiftEvalPretrain"
SMALL_MB = 400
FAMILY_KEEP = {"cmip6": 4, "era5": 4, "largest": 2}
MAX_LEN = 16384
MAX_SERIES_PER_SOURCE = 60000


def plan_files(budget_gb: float) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    files = [f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True) if hasattr(f, "size")]
    by_dir: dict[str, list] = defaultdict(list)
    for f in files:
        if "/" in f.path:
            by_dir[f.path.split("/")[0]].append(f)
    fam_seen: dict[str, int] = defaultdict(int)
    chosen, total = [], 0
    for d in sorted(by_dir):
        fam = re.sub(r"_\d{4}$", "", d)
        if fam in FAMILY_KEEP:
            fam_seen[fam] += 1
            # spread the kept members across the family's years
            members = sorted(x for x in by_dir if re.sub(r"_\d{4}$", "", x) == fam)
            step = max(1, len(members) // FAMILY_KEEP[fam])
            if d not in members[::step][: FAMILY_KEEP[fam]]:
                continue
        fs = by_dir[d]
        # skip HF `cache-*.arrow` files (some are 38 GB); real shards are `data-*.arrow`
        arrows = sorted([f for f in fs if f.path.split("/")[-1].startswith("data-") and f.path.endswith(".arrow")],
                        key=lambda f: f.path)
        meta = [f for f in fs if f.path.endswith(".json")]
        size = sum(f.size for f in arrows)
        take = arrows if size <= SMALL_MB * 1e6 else [min(arrows, key=lambda f: f.size)]
        chosen += [f.path for f in take + meta]
        total += sum(f.size for f in take)
    print(f"planned {len(chosen)} files, {total / 1e9:.2f} GB")
    if total > budget_gb * 1e9:
        raise SystemExit(f"over budget ({budget_gb} GB); lower SMALL_MB")
    return chosen


def iter_series(arrow_path: Path):
    import pyarrow as pa

    with pa.memory_map(str(arrow_path)) as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(src).read_all()
    col = table.column("target")
    freqs = table.column("freq").to_pylist() if "freq" in table.column_names else [None] * len(col)
    for i in range(len(col)):
        t = col[i].as_py()
        if t is None or len(t) == 0:
            continue
        if isinstance(t[0], list):
            for ch in t:
                yield np.asarray(ch, dtype=np.float32), freqs[i]
        else:
            yield np.asarray(t, dtype=np.float32), freqs[i]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=7.0)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--raw", default="data/pretrain_raw")
    ap.add_argument("--out", default="data/pretrain")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    files = plan_files(args.budget_gb)
    snapshot_download(REPO, repo_type="dataset", local_dir=args.raw, allow_patterns=files, max_workers=8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    values = open(out / "values.f32", "wb")
    index, sources, src_stats = [], [], {}
    offset = 0
    for d in sorted(p for p in Path(args.raw).iterdir() if p.is_dir()):
        sid = len(sources)
        sources.append(d.name)
        n_ser = n_pts = 0
        freq = None
        for arrow in sorted(d.glob("*.arrow")):
            for x, f in iter_series(arrow):
                freq = freq or f
                if n_ser >= MAX_SERIES_PER_SOURCE:
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
                n_ser += 1
                n_pts += len(x)
        src_stats[d.name] = {"series": n_ser, "points": n_pts, "freq": freq}
        print(f"{d.name:40s} series={n_ser:7d} points={n_pts:12,d} freq={freq}", flush=True)
    values.close()
    np.save(out / "index.npy", np.asarray(index, dtype=np.int64))
    json.dump({"sources": sources, "stats": src_stats, "total_points": offset}, open(out / "meta.json", "w"), indent=1)
    print(f"packed {len(index)} series, {offset:,d} points ({offset * 4 / 1e9:.2f} GB)")
    if not args.keep_raw:
        shutil.rmtree(args.raw)


if __name__ == "__main__":
    main()
