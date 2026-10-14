"""A/B: all boxes vs only the boxes within the arm's reach, same stations, same seeds. Needs cuRobo + GPU.

  python validation/bench_near.py <maps_root> <site> [object_index] [start_r_min start_r_max]
"""
import json
import sys
from pathlib import Path
import time

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/validation")
from robot_model import load_config  # noqa: E402
import scene_io  # noqa: E402
from bench_lab_maps import load_site, start_pose  # noqa: E402
from bench_multi_grasp import make_grasps, build, candidates, score_pool, sync  # noqa: E402

root, site = sys.argv[1], sys.argv[2]
oi = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rmin, rmax = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (0.9, 1.3)
cfg = load_config(REPO + "/config.yaml")
data, objs, occ = load_site(f"{root}/{site}")
o = objs[oi]
start = start_pose(occ, np.array(o["pos"][:2]), rmin, rmax)
scene = scene_io.scene_from_slam(data, start, o["pos"], [1.0, 0.0, 0.0, 0.0], cfg, target=o["target"], occupancy=occ)
grasps = make_grasps(scene, 20, pitches_deg=(0, 45))
label = f"{site}/{o['cls']}#{o['inst']}"
print(f"{label}: {len(scene.boxes_base)} boxes, start {np.round(start, 2).tolist()}, "
      f"target {np.hypot(*scene.grasp_pos_base[:2]):.2f} m away", flush=True)

SEEDS = 16


def run(tag, near, group, pool=None):
    torch.manual_seed(0); np.random.seed(0)
    pl, setup = build(cfg, scene, **{"profile.scorer_seeds": SEEDS, "profile.scorer_group": group,
                                     "profile.near_boxes_only": near})
    if pool is None:
        t = time.perf_counter()
        pool = [(gi, st) for gi, g in enumerate(grasps) for st in candidates(pl, g, 3, stop_on_zero=True)]
        sync(); print(f"  pool: {len(pool)} stations from {len({g for g, _ in pool})} grasps, ik {time.perf_counter() - t:.1f}s", flush=True)
    torch.manual_seed(0)
    sync(); t = time.perf_counter()
    r = np.array(score_pool(pl, [(grasps[gi], st) for gi, st in pool]))
    sync(); dt = time.perf_counter() - t
    # boxes each env actually holds
    F = np.array([st[:3] for _, st in pool]); envf = (F[:, None, :] + pl.OFFS[None]).reshape(-1, 3)
    n_env = pl._near(envf).sum(1)
    print(f"  {tag:<28} boxes kept {int(pl.T['boxes_used'])}, cache {int(pl.T['near_cache'])}, per env median "
          f"{np.median(n_env):.0f} max {n_env.max()}, group {pl.group}, setup {setup:.1f}s, score {dt:.1f}s "
          f"({1000 * dt / len(pool):.0f} ms/station)", flush=True)
    out = {"tag": tag, "setup_s": setup, "score_s": dt, "group": pl.group, "boxes_used": pl.T["boxes_used"],
           "cache": pl.T["near_cache"], "env_median": float(np.median(n_env)), "env_max": int(n_env.max()), "rob": r.tolist()}
    del pl; torch.cuda.empty_cache()
    return pool, out


pool, A1 = run("all boxes (before) #1", False, 2)
A = [A1] + [run(f"all boxes (before) #{i}", False, 2, pool)[1] for i in (2, 3)]
B = [run(f"near boxes, group 2 #{i}", True, 2, pool)[1] for i in (1, 2, 3)]
G4 = [run(f"near boxes, group 4 #{i}", True, 4, pool)[1] for i in (1, 2)]
rA, rB, rG = (np.array([x["rob"] for x in L]) for L in (A, B, G4))
def pairs(X, Y, same):
    return [np.abs(X[i] - Y[j]) for i in range(len(X)) for j in range(len(Y)) if not (same and j <= i)]
def pick(r):
    return min(range(len(pool)), key=lambda i: (-r[i], float(np.hypot(*pool[i][1][:2]))))
def report(name, X, Y, same):
    d = np.array(pairs(X, Y, same)); th = cfg["robustness"]["threshold"]; fl = cfg["robustness"]["min_accept"]
    agree = lambda t: np.mean([((x >= t - 1e-9) == (y >= t - 1e-9)).mean() for i, x in enumerate(X) for j, y in enumerate(Y) if not (same and j <= i)])
    picks = np.mean([pick(x) == pick(y) for i, x in enumerate(X) for j, y in enumerate(Y) if not (same and j <= i)])
    print(f"  {name:<34} mean|diff| {d.mean():.3f}  max {d.max():.3f}  pass(>={th}) agree {agree(th):.3f}  "
          f"accept(>={fl}) agree {agree(fl):.3f}  same pick {picks:.2f}", flush=True)
print(f"  mean score: before {rA.mean():.3f}, near/g2 {rB.mean():.3f}, near/g4 {rG.mean():.3f}", flush=True)
report("before vs before (cuRobo noise)", rA, rA, True)
report("before vs near, group 2", rA, rB, False)
report("before vs near, group 4", rA, rG, False)
report("near vs near (noise)", rB, rB, True)
ts = lambda L: (np.mean([x["score_s"] for x in L]), np.mean([x["setup_s"] for x in L]))
print(f"  time: before score {ts(A)[0]:.1f}s setup {ts(A)[1]:.1f}s | near/g2 score {ts(B)[0]:.1f}s setup {ts(B)[1]:.1f}s "
      f"| near/g4 score {ts(G4)[0]:.1f}s setup {ts(G4)[1]:.1f}s  -> {ts(A)[0] / ts(G4)[0]:.1f}x faster scoring", flush=True)
print("@@JSON@@" + json.dumps({"scene": label, "n_boxes": len(scene.boxes_base), "pool": [[gi, st.tolist()] for gi, st in pool],
                               "runs": A + B + G4}))
