# Model card — Alma-WM (alma_wm1)
- **Intended use:** research. Zero-shot probabilistic forecasting of univariate series, and
  action-conditioned "what-if" rollouts for systems whose inputs are known.
- **Out of scope:**
  - Trading or financial decisions.
  - Safety-critical control.
  - Causal claims about real systems: actions were validated on simulated systems only.
  - Multivariate joint forecasting.
- **Data:** GiftEvalPretrain (94 sources, 1.20B points), TSMixup, synthetic GP/structural series, and
  5 simulated controlled-system families.
- **Known limits:**
  - Marginal quantiles only (no joint sample paths).
  - At most 2 action channels.
  - Real-world action effects are untested.
  - One training seed.
- **Owner:** the project owner (user). Built and validated by Claude (not independent).
- **Retirement criteria:** a successor beats it on a new sealed window and re-passes the gates (14.5),
  or any kill trigger in preregistration.md fires.
