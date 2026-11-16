"""Shared helpers for the Alma-WM validation harness."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

RUN = ROOT / os.environ.get("VALRUN", (ROOT / "validation_runs/CURRENT").read_text().strip())
QS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
CTX, HZ = 2048, 512  # simulated episodes: context and future length
PATCH = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(RUN / "report/running_log.md", "a") as f:
        f.write(line + "\n")


def manifest_add(section: str, path: Path, extra: dict | None = None) -> None:
    mf = RUN / "manifest.json"
    m = json.loads(mf.read_text()) if mf.exists() else {}
    m.setdefault(section, {})[str(path.relative_to(ROOT))] = {
        "sha256": sha256(path), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})}
    mf.write_text(json.dumps(m, indent=1))


def seal(name: str, **arrays) -> Path:
    """Write predictions and register their hash + time BEFORE any truth is loaded."""
    p = RUN / "sealed_outputs" / f"{name}.npz"
    np.savez_compressed(p, **arrays)
    manifest_add("sealed", p)
    return p


def load_sealed(name: str) -> dict:
    p = RUN / "sealed_outputs" / f"{name}.npz"
    m = json.loads((RUN / "manifest.json").read_text())
    rec = m.get("sealed", {}).get(str(p.relative_to(ROOT)))
    if rec is None or rec["sha256"] != sha256(p):
        raise RuntimeError(f"refusing to grade unsealed or modified output: {p}")
    truth_t = m.get("truth_loaded", {}).get(name)
    if truth_t and truth_t < rec["utc"]:
        raise RuntimeError(f"truth for {name} was loaded before sealing")
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_truth(name: str) -> dict:
    mf = RUN / "manifest.json"
    m = json.loads(mf.read_text())
    m.setdefault("truth_loaded", {})[name] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mf.write_text(json.dumps(m, indent=1))
    z = np.load(RUN / "harness" / f"truth_{name}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------------------------


def wql(y: np.ndarray, q: np.ndarray, norm: float | None = None) -> float:
    """Weighted quantile loss (CRPS approximation). y: (H,), q: (H, 9)."""
    d = y[:, None] - q
    loss = 2 * np.maximum(QS * d, (QS - 1) * d).sum(-1).mean() / len(QS)
    den = norm if norm is not None else np.abs(y).mean()
    return float(loss / max(den, 1e-8))


def coverage(y: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    c80 = np.mean((y >= q[:, 0]) & (y <= q[:, 8]))
    c60 = np.mean((y >= q[:, 1]) & (y <= q[:, 7]))
    return float(c80), float(c60)


def paired_t(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t on log(a/b) per unit. Negative mean = a better (lower loss)."""
    r = np.log(np.maximum(a, 1e-12) / np.maximum(b, 1e-12))
    r = r[np.isfinite(r)]
    n = len(r)
    sd = r.std(ddof=1) if n > 1 else np.nan
    t = r.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    rng = np.random.default_rng(0)
    bs = np.array([r[rng.integers(0, n, n)].mean() for _ in range(2000)]) if n > 1 else np.array([np.nan])
    return {"n": int(n), "gmean_ratio": float(np.exp(r.mean())), "t": float(-t) if np.isfinite(t) else None,
            "ci95_ratio": [float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5)))],
            "win_rate": float(np.mean(r < 0))}


def holm(pvals: dict, alpha: float = 0.05) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, still = len(items), {}, True
    for i, (k, p) in enumerate(items):
        ok = still and p <= alpha / (m - i)
        still = ok
        out[k] = bool(ok)
    return out


def t_to_p(t: float, n: int) -> float:
    from scipy import stats

    return float(stats.t.sf(t, n - 1)) if t is not None and n > 1 else 1.0
