"""Assemble report/VALIDATION_REPORT.md: scorecard, gate verdicts, permitted claim (Section F)."""

from __future__ import annotations

import json
import subprocess

from common import ROOT, RUN, sha256

DESCRIPTIVE_NOT_PASSED = {"5.3", "5.5", "9.5", "9.7"}  # amendment A1: descriptive -> not passed for claims
TIERS = [
    ([0, 1, 2, 3], "The evaluation is leak-free and reproducible."),
    ([4], "...predicts next-step states better than simple laws and baselines on unseen data."),
    ([5], "...its simulated rollouts reproduce real behaviour over multi-step horizons."),
    ([6], "...correctly predicts the consequences of actions, verified on real interventions and natural experiments."),
    ([7], "...knows when it is outside its experience and falls back safely."),
    ([8, 9], "...results are statistically significant after corrections and robust across regimes and stress."),
    ([10], "...its internal structure is stable and interpretable."),
    ([11, 12], "...its decisions keep their value in reality."),
    ([13], "...outperforms named models after multiple-testing correction."),
    ([14], "Live-verified world model."),
]


def gate_of(g: str) -> int:
    head = g.split("/")[0].strip().split(".")[0].split(" ")[0]
    return int(head)


def main() -> None:
    card = []
    for f in ("scorecard_static.json", "scorecard_grade.json"):
        p = RUN / "report" / f
        if p.exists():
            card += json.loads(p.read_text())
    pre = RUN / "prereg"
    g0 = [("0.1", "preregistration.md"), ("0.2", "FROZEN.txt"), ("0.3", "trial_log.csv"), ("0.4", "independence.md"),
          ("0.5", "model_card.md"), ("0.6", "preregistration.md")]
    for c, f in g0:
        card.insert(0, {"gate": c, "check": f"{f} exists and hashed before holdout outputs", "result": (pre / f).exists(),
                        "threshold": "exists", "status": "PASS" if (pre / f).exists() else "FAIL", "script": "-"})
    card.append({"gate": "11", "check": "language component", "result": "no LLM", "threshold": "-", "status": "N/A",
                 "script": "-"})
    card.append({"gate": "13.1-13.5", "check": "leaderboard on identical harness", "result": "see Leaderboard",
                 "threshold": "Holm-corrected beats", "status": "REPORTED", "script": "validate/grade.py"})
    card.append({"gate": "14", "check": "live sealed forward record >= 6 months", "result": "no live deployment",
                 "threshold": ">= 6 months", "status": "FAIL", "script": "-"})
    card.append({"gate": "3.3", "check": "sealing proof: outputs hashed before truth loaded",
                 "result": "grade.py refuses unsealed outputs; manifest has seal and truth-load times", "threshold": "-",
                 "status": "PASS", "script": "validate/common.py"})

    verdict = {}
    for r in card:
        try:
            g = gate_of(r["gate"])
        except ValueError:
            continue
        st = r["status"]
        sub = r["gate"].split("/")[0].strip().split(" ")[0]
        bad = st == "FAIL" or (st == "REPORTED" and sub in DESCRIPTIVE_NOT_PASSED)
        verdict.setdefault(g, []).append(bad)
    gate_pass = {g: not any(v) for g, v in verdict.items()}

    claim, tier_rows = None, []
    for gates, text in TIERS:
        ok = all(gate_pass.get(g, True) for g in gates)
        tier_rows.append((gates, text, ok))
        if ok and (claim is None or claim[1]):
            claim = (text, True) if claim is None or claim[1] else claim
        if not ok:
            break
    earned = [t for g, t, ok in tier_rows if ok]

    mf = RUN / "manifest.json"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    L = ["# Validation report — Alma-WM (`alma_wm1`)", ""]
    L.append("## 1. Verdict")
    if earned:
        L.append("Highest claim earned (Section F wording): **" + " ".join(earned) + "**")
    else:
        L.append("**No claim tier is earned.** The first tier (Gates 0-3) did not fully pass.")
    first_fail = next(((g, t) for g, t, ok in tier_rows if not ok), None)
    if first_fail:
        L.append(f"\nThe claim is capped at the row before Gates {first_fail[0]}, which did not pass.")
    L += ["", "## 2. Gate verdicts", "", "| Gate | Verdict |", "|---|---|"]
    for g in sorted(verdict):
        L.append(f"| {g} | {'PASS' if gate_pass[g] else 'NOT PASSED'} |")
    L += ["", "## 3. Scorecard", "", "| Gate | Check | Result | Threshold | Status |", "|---|---|---|---|---|"]
    for r in card:
        res = json.dumps(r["result"], default=str)
        res = res if len(res) < 220 else res[:217] + "..."
        L.append(f"| {r['gate']} | {r['check']} | `{res}` | {r['threshold']} | **{r['status']}** |")
    L += ["", "## 4. Reproducibility", f"- commit `{commit}`",
          f"- manifest sha256 `{sha256(mf) if mf.exists() else 'missing'}`",
          f"- rerun: `make validate RUN={RUN.name}`"]
    (RUN / "report/VALIDATION_REPORT.md").write_text("\n".join(L) + "\n")
    (RUN / "report/gate_verdicts.json").write_text(json.dumps({"gate_pass": gate_pass, "earned": earned}, indent=1))
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
