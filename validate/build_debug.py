"""Debug suites (never graded as evidence): fresh seeds + burned GIFT development windows."""
import glob

import numpy as np

from common import RUN, log, manifest_add
import build_suite as B
import simulators as S

assert "DEBUG" in str(RUN), RUN
B.build_sim("sim_val", range(700000, 700060), S.TRAIN_KINDS, with_mc=False)
B.build_sim("sim_test", range(710000, 710060), S.TRAIN_KINDS)
B.build_sim("sim_tanks", range(720000, 720020), S.HELDOUT_KINDS)
B.build_planning("plan_test", range(730000, 730020), ("thermal", "fopdt", "lss", "pricing"))
C, L, META = [], [], []
for p in sorted(glob.glob("data/gift_cache/*__1000.npz"))[:14]:
    z = np.load(p, allow_pickle=True)
    nm = p.split("/")[-1].replace("__1000.npz", "")
    for i in range(10):
        C.append(np.asarray(z["contexts"][i], np.float32)); L.append(z["labels"][i]); META.append((nm, int(z["H"]), int(z["season"])))
np.savez_compressed(RUN / "harness/inputs_real.npz", contexts=np.array(C, dtype=object), meta=np.array(META, dtype=object))
np.savez_compressed(RUN / "harness/truth_real.npz", labels=np.array(L, dtype=object))
for f in ("inputs", "truth"):
    manifest_add("harness", RUN / f"harness/{f}_real.npz")
log("debug suites built")
