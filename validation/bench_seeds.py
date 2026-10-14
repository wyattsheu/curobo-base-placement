"""How many scorer seeds (and is a warm start worth it) once far boxes are dropped? Same stations as a bench_near.py run.
Reference: 32 seeds, one station per call. Needs cuRobo + GPU.

  python validation/bench_seeds.py <maps_root> <site> <bench_near log>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/validation")
from robot_model import load_config  # noqa: E402
import scene_io  # noqa: E402
from bench_lab_maps import load_site, start_pose  # noqa: E402
from bench_multi_grasp import make_grasps, build, score_pool, sync  # noqa: E402

root, site, log = sys.argv[1:4]
row = json.loads(next(l for l in open(log) if l.startswith("@@JSON@@"))[8:])
cfg = load_config(REPO + "/config.yaml")
data, objs, occ = load_site(f"{root}/{site}"); o = objs[0]
scene = scene_io.scene_from_slam(data, start_pose(occ, np.array(o["pos"][:2])), o["pos"], [1, 0, 0, 0], cfg,
                                 target=o["target"], occupancy=occ)
grasps = make_grasps(scene, 20, pitches_deg=(0, 45))
pool = [(grasps[gi], np.array(st)) for gi, st in row["pool"]]
floor, th = cfg["robustness"]["min_accept"], cfg["robustness"]["threshold"]


def pick(r):
    return min(range(len(pool)), key=lambda i: (-r[i], math.hypot(*pool[i][1][:2])))


def run(seeds, group, warm):
    pl, _ = build(cfg, scene, **{"profile.scorer_seeds": seeds, "profile.scorer_group": group})
    score_pool(pl, pool[:pl.group], warm)                        # warm-up at this batch shape
    sync(); t = time.perf_counter()
    r = np.array(score_pool(pl, pool, warm)); sync()
    out = (r, time.perf_counter() - t, pl.group)
    del pl; torch.cuda.empty_cache()
    return out


ref, tref, _ = run(32, 1, False)
best = ref.max()
print(f"{site}: reference 32 seeds / 1 station per call: mean {ref.mean():.3f}, {tref:.1f}s", flush=True)
rows = []
for seeds, warm in ((16, False), (8, False), (4, False), (8, True), (4, True), (2, True), (1, True)):
    r, t, g = run(seeds, 4, warm)
    k = pick(r)
    d = {"seeds": seeds, "warm": warm, "group": g, "score_s": t, "mean": float(r.mean()), "regret": float(best - ref[k]),
         "mae": float(np.abs(r - ref).mean()), "pass_agree": float(((r >= th - 1e-9) == (ref >= th - 1e-9)).mean()),
         "accept_agree": float(((r >= floor) == (ref >= floor)).mean()), "same_pick": k == pick(ref)}
    rows.append(d)
    print(f"  seeds {seeds:2d} warm {str(warm):5}  {t:5.1f}s  mean {d['mean']:.3f}  |diff| {d['mae']:.3f}  regret {d['regret']:.3f}  "
          f"pass agree {d['pass_agree']:.2f}  accept agree {d['accept_agree']:.2f}  same pick {d['same_pick']}", flush=True)
print("@@JSON@@" + json.dumps({"site": site, "ref_mean": float(ref.mean()), "ref_s": tref, "rows": rows}))
