# Overnight notes 2026-11-19
- Stage D: +1.5B new points, 3k-step small A/B: F 1.0012, G 1.0028 vs controls (neutral). D2 skipped by rule.
- Honest 2-fold ensemble search (12 members): global cv 0.975 all / 1.021 sealed28; bucket-routed 0.971 / 1.027 (overfits sealed).
- Stage E v3new (both corpora, bucket+mix) @2k: 0.9816 all / 1.0323 sealed28 vs v3@10k 0.9809 / 1.0334 -> neutral so far.
  Training ~8 s/step (memory pressure), 10k steps ETA ~Nov 20 afternoon.
- Sealed28 gap (ens v3+D vs Telos): CRPS 1.017, MASE 1.027 -> the gap is POINT accuracy, not calibration.
  Worst: bitbrains_fast_storage_5T long 1.19, m4_yearly 1.15, LOOP_SEATTLE_H 1.11, m4_quarterly 1.08, LOOP_SEATTLE_5T x3 ~1.06,
  bitbrains_rnd_5T medium 1.07. Families: LOOP_SEATTLE (traffic) and bitbrains (cloud) cover 9 of the 18 losing sealed tasks.
- Next candidates: traffic/cloud-heavy sampling or synthetic priors for spiky/regime series; short-series (yearly/quarterly) point model.
- 01:55 Stage E stopped: v3new 4k 0.9832/1.0353 (worse than 2k). More-data continuation abandoned.
- Per-family (fam_report.py): every model loses LOOP_SEATTLE (~1.06-1.09), m4 (1.07-1.12), M_DENSE (1.18-1.19). Systemic.
- Gap is largest in the FIRST quarter of the horizon (1.19-1.40 on losing tasks), i.e. point accuracy right after the context.
- symmetric=False: worse overall (0.997 vs 0.984 all) -> keep symmetric; D_nosym kept as a diverse ensemble member.
- Post-hoc per-bucket spread calibration (2-fold): no gain (+-0.0003) -> widths are fine; the problem is the point forecast.
- Stage F (running): D recipe + traffic/cloud source boost x3/x5, alpha 0.3, seed replicate. Stage G (queued): v3 fine-tune with best boost if it beats D by >0.5%.

## PAUSED 2026-11-20 09:1x (user request)
Saved and complete:
- Stage F: H_boost (0.9854/1.0210), H_seed (0.9866/1.0207) trained + scored (results/preds/*.json/.npz).
- H_boost5 trained to 3000 (runs/stageF/H_boost5/ema_3000.pt); dev2 not yet scored.
- All earlier checkpoints: runs/alma_v3/, runs/stageA..E/; predictions for 17 models in results/preds/.
Not started: H_a03 arm, ensemble search F, Stage G (v3 boost fine-tune, only if a boost arm beats D by 0.5%).
Resume: ./resume_all.sh  (skips everything already done; keeps the Mac awake via caffeinate)
Key finding: the sealed gap (LOOP_SEATTLE_H 1.10, M_DENSE 1.22, m4 yearly/quarterly 1.07-1.15) is the same across all
data/recipe/seed variants -> needs an architecture change, not more data.
