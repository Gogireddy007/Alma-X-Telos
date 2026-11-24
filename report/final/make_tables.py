"""Emit LaTeX table bodies from numbers.json (no hand-typed numbers)."""
import json
from pathlib import Path
N = json.loads(Path("report/final/numbers.json").read_text())
esc = lambda s: s.replace("_", r"\_")
out = []
rows = sorted(N["sealed_per_task"].items(), key=lambda kv: kv[0])
out.append("% sealed per-task\n\\newcommand{\\SealedRows}{%")
for t, d in rows:
    b = lambda x: f"\\textbf{{{x:.3f}}}"
    vals = {"v3": d["v3"], "v2": d["v2"], "telos": d["telos"]}
    best = min(vals, key=vals.get)
    cell = {k: (b(v) if k == best else f"{v:.3f}") for k, v in vals.items()}
    out.append(f"{esc(t)} & {cell['v3']} & {cell['v2']} & {cell['telos']} & {d['v3_over_telos']:.3f} & {d['v3_over_v2']:.3f} \\\\")
out.append("}")
out.append("\\newcommand{\\DatasetRows}{%")
for f, d in sorted(N["gift_by_dataset"].items(), key=lambda kv: kv[1]["v3_over_telos"]):
    out.append(f"{esc(f)} & {d['n']} & {d['v3_over_telos']:.3f} & {d['v3_over_v2']:.3f} \\\\")
out.append("}")
Path("report/final/tables.tex").write_text("\n".join(out) + "\n")
print("rows", len(rows), len(N["gift_by_dataset"]))
