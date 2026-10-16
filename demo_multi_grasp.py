"""Example of the whole flow with N candidate grasps: where to drive for each, which ones are impossible, which one to take.

    python demo_multi_grasp.py --lab-map <map dir> --run <Grasp Lab run dir> [--top 8] [--first] [--drop-stale]

Times in the comments were measured on the Jetson Thor (120 W, GPU shared with the Qwen / Molmo containers,
map803_0930 + run 20261008_045944_bottle, 8 GraspGen grasps + the executed one + one impossible); the script prints
its own as it goes.

  1. startup, once per process     planner = BasePlacementPlanner(cfg)       ~14 s (all cuRobo solvers + warm-up)
                                   slam = load_lab_map(map_dir)               ~3.8 s the first time a map is ever seen,
                                                                              ~7 ms afterwards (.map_cache/)
  2. once per wrist capture        world = merge(slam, capture)               ~0.19 s (here grasp_lab.py; once the
                                                                              teammate delivers merged maps, read those)
  3. once per candidate grasp      solve(...)                                 ~0.18 s scene (boxes + ESDF; the very first
                                                                              one ~0.6 s: box-fitting workers start)
                                                                              + plan(): 0.3–0.8 s with a 27/27 station,
                                                                              ~1.5 s if impossible, 5–7 s if only a weaker
                                                                              station exists (fallback / not_robust)
  4. pick one                      pick(results)                              (a simple rule, to be discussed)

All 10 grasps one after another took 30 s on the Thor, mostly the fallback ones. --first tries them in upstream's
confidence order and stops at the first one with a 27/27 station (here: the first grasp, ~1 s).

A grasp that cannot be done comes back with ok = False and the reason (see README §2.3):
  no_ik_solution  the gripper pose is out of reach, or every way to reach it collides
  outside_cone    reachable only with the target outside the arm's ±30° safety cone
  cannot_park     every base pose that reaches it is blocked or off the drivable area
  not_robust      reachable, but no base pose survives at least min_accept of the navigation errors
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from robot_model import load_config
import scene_io
import grasp_lab
from lab_map import load_lab_map

HERE = Path(__file__).parent


# ---------------------------------------------------------------------------- 1. startup (once)
def start(cfg):
    """All cuRobo solvers, built and warmed up once; every request after this only writes obstacles into them."""
    from placement import BasePlacementPlanner                               # needs cuRobo + GPU
    return BasePlacementPlanner(cfg)                                          # Thor ~10 s, once


# ---------------------------------------------------------------------------- 2. per capture
def merge(cfg, slam, names, run_dir, map_dir, drop_stale=False):
    """The global SLAM map with what the wrist camera saw at this capture (grasp_lab.fuse), kept in memory.
    → (world dict for scene_from_slam, capture)."""
    cap = grasp_lab.load_capture(run_dir, float(slam["floor_height_m"]), map_dir)   # ~20 ms
    return grasp_lab.fuse(cfg, slam, cap, names, drop_stale), cap                   # ~0.1 s


# ---------------------------------------------------------------------------- 3. per grasp
def solve(planner, cfg, world, base_pose, grasp, occupancy):
    """One candidate grasp (gripper pose in the map frame) → where Kachaka should go, or why it cannot."""
    t0 = time.perf_counter()
    scene = scene_io.scene_from_slam(world, base_pose, grasp["position"], grasp["quaternion_wxyz"], cfg,
                                     target={"mask_key": "target_mask"}, occupancy=occupancy)   # ~0.2 s
    t1 = time.perf_counter()
    planner.set_scene(scene)
    r = planner.plan()                                                        # ~0.5–1 s when a 27/27 station exists
    t2 = time.perf_counter()
    out = {"ok": r["ok"], "reason": r["reason"], "below_threshold": r["below_threshold"], "robustness": r["robustness"],
           "chosen_map": r.get("chosen_map"), "scene_s": t1 - t0, "plan_s": t2 - t1}
    if r["ok"]:
        x, y, _ = r["chosen_map"]
        out["drive_m"] = math.hypot(x - base_pose[0], y - base_pose[1])
    return out


# ---------------------------------------------------------------------------- 4. pick one
def pick(results):
    """A simple rule (to be discussed with upstream): a station that survives every navigation error first, then the
    most robust, then the shortest drive, then upstream's own grasp confidence. → index or None."""
    ok = [i for i, r in enumerate(results) if r["ok"]]
    if not ok:
        return None
    return min(ok, key=lambda i: (results[i]["below_threshold"], -results[i]["robustness"], results[i]["drive_m"],
                                  -(results[i].get("confidence") or 0.0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--lab-map", required=True)
    ap.add_argument("--run", required=True, help="Grasp Lab run folder (the wrist capture and its GraspGen grasps)")
    ap.add_argument("--top", type=int, default=8, help="GraspGen grasps to try besides the executed one")
    ap.add_argument("--drop-stale", action="store_true", help="test data only, see grasp_lab.py")
    ap.add_argument("--first", action="store_true", help="stop at the first grasp (upstream's order) with a 27/27 station")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    T = {}

    # 1. startup -------------------------------------------------------------------------------------------------
    t = time.perf_counter(); planner = start(cfg); T["startup: planner"] = time.perf_counter() - t
    t = time.perf_counter(); slam, names, _ = load_lab_map(a.lab_map); T["startup: SLAM map"] = time.perf_counter() - t
    occ = next(iter(sorted(Path(a.lab_map).glob("occupancy_*.yaml"))), None)
    occ = str(occ) if occ else None

    # 2. a wrist capture arrives -----------------------------------------------------------------------------------
    t = time.perf_counter(); world, cap = merge(cfg, slam, names, a.run, a.lab_map, a.drop_stale); T["capture: merge"] = time.perf_counter() - t
    base_pose = grasp_lab.kachaka_pose(cfg, cap["map_from_arm"])               # where Kachaka stands now
    grasps = [{"name": r["grasp"], "confidence": r["confidence"], **r["grasp_pose_in_map"]}
              for r in grasp_lab.requests(cfg, cap, a.top)]
    # one more that cannot be done, to show the rejection: the same gripper pose 2 m up in the air
    g0 = grasps[0]
    grasps.append({"name": "impossible (2 m up)", "confidence": None, "quaternion_wxyz": g0["quaternion_wxyz"],
                   "position": [g0["position"][0], g0["position"][1], g0["position"][2] + 2.0]})

    # 3. every candidate grasp -------------------------------------------------------------------------------------
    results = []
    for g in grasps:
        r = solve(planner, cfg, world, base_pose, g, occ)
        r.update(name=g["name"], confidence=g["confidence"]); results.append(r)
        if r["ok"]:
            x, y, yaw = r["chosen_map"]
            verdict = (f"drive to ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f}°), {r['drive_m']:.2f} m away, "
                       f"robustness {r['robustness'] * 27:.0f}/27" + (" (fallback)" if r["below_threshold"] else ""))
        else:
            verdict = f"REJECTED: {r['reason']}"
        print(f"{g['name']:22s} {verdict:60s} scene {r['scene_s']:.2f} s + plan {r['plan_s']:.2f} s", flush=True)
        if a.first and r["ok"] and not r["below_threshold"]:
            break                                                               # good enough: stop trying

    # 4. pick one ------------------------------------------------------------------------------------------------------
    k = pick(results)
    print("\npicked:", "none of them can be done" if k is None else
          f"{results[k]['name']} → drive to {tuple(round(v, 3) for v in results[k]['chosen_map'])}")
    per = [r["scene_s"] + r["plan_s"] for r in results]
    print(f"\nstartup: planner {T['startup: planner']:.1f} s, SLAM map {T['startup: SLAM map']:.3f} s (once) | "
          f"capture merge {T['capture: merge']:.2f} s | {len(results)} grasps: {sum(per):.1f} s in all, "
          f"median {np.median(per):.2f} s each")
    print("@@JSON@@" + json.dumps({"results": results, "picked": k, "startup_s": T}, default=float))


if __name__ == "__main__":
    main()
