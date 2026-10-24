"""Fewer navigation-error checks (robustness.perturb_set) and stopping the whole-body IK early when its first round finds
nothing (profile.wb_stop_if_empty): time and chosen station against today, on the same requests. Needs cuRobo + GPU.

  python validation/bench_perturb.py --lab <maps_root> <site> [<site> ...]    # bench_roi's requests (as profile_plan.py)
  python validation/bench_perturb.py --requests requests.jsonl                # e.g. grasp_lab.py output

Settings (one long-running planner each, every request, same scenes):
  full27            today: every (-, 0, +) of x, y and yaw
  full27 reversed   today with the requests in reverse order: cuRobo's own random state differs, the yardstick for
                    "a different station was chosen" (seed luck, not the change)
  full27+stop       today + stop the whole-body IK after an empty first round
  xy9, xy5          position only: the 3 x 3 grid / the centre and 4 corners, yaw error not checked
  xy5+stop          xy5 + early stop
  corners9          the centre and the 8 corners of the x/y/yaw box (the older 9-point option)
  corners9+stop     corners9 + early stop
Every station a reduced set chooses is re-scored with all 27 checks by the full27 planner: does it really survive
every navigation error, yaw included?
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from robot_model import load_config  # noqa: E402
from bench_bound import load_requests  # noqa: E402

SETTINGS = [("full27", "full27", False), ("full27 reversed", "full27", False), ("full27+stop", "full27", True),
            ("xy9", "xy9", False), ("xy5", "xy5", False), ("xy5+stop", "xy5", True), ("corners9", "corners9", False),
            ("corners9+stop", "corners9", True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests"); ap.add_argument("--lab", nargs="+")
    ap.add_argument("--settings", nargs="+", help="subset of: " + ", ".join(t for t, _, _ in SETTINGS))
    a = ap.parse_args()
    from placement import BasePlacementPlanner
    cfg0 = load_config(HERE.parent / "config.yaml")
    t = time.perf_counter()
    scenes = []
    for n, f in load_requests(a, cfg0):                              # same scenes for every setting
        t0 = time.perf_counter(); sc = f(cfg0); scenes.append((n, sc, time.perf_counter() - t0))
    print(json.dumps({"scenes": len(scenes), "build_s": time.perf_counter() - t}), flush=True)
    rows, chosen = [], {}
    for tag, pset, stop in [x for x in SETTINGS if not a.settings or x[0] in a.settings]:
        cfg = copy.deepcopy(cfg0); cfg["robustness"]["perturb_set"] = pset; cfg["profile"]["wb_stop_if_empty"] = stop
        torch.cuda.empty_cache()
        pl = BasePlacementPlanner(cfg)
        for name, sc, scene_s in (scenes[::-1] if "reversed" in tag else scenes):
            t0 = time.perf_counter(); pl.set_scene(sc); r = pl.plan(); dt = time.perf_counter() - t0
            T = r["timing_s"]
            row = {"req": name, "setting": tag, "ok": r["ok"], "reason": r["reason"], "below": r["below_threshold"],
                   "robustness": r["robustness"], "chosen": r.get("chosen_base"), "n_checked": r["n_checked"],
                   "wb_rounds": r["wb_rounds"], "n_perturb": r["n_perturb"], "scene_s": scene_s, "plan_s": dt,
                   "wholebody_ik_s": T["wholebody_ik_s"], "scoring_s": T["scoring_s"]}
            rows.append(row); print(json.dumps(row), flush=True)
            if r["ok"]:
                chosen[(name, tag)] = r["chosen_base"]
        del pl
    # every chosen station re-scored with all 27 checks
    cfg = copy.deepcopy(cfg0); cfg["robustness"]["perturb_set"] = "full27"
    pl = BasePlacementPlanner(cfg)
    for name, sc, _ in scenes:
        keys = [k for k in chosen if k[0] == name]
        if not keys:
            continue
        pl.set_scene(sc)
        for k in keys:
            s27 = float(pl.robustness(np.array([chosen[k]]))[0])
            for row in rows:
                if row["req"] == name and row["setting"] == k[1]:
                    row["score27_of_choice"] = s27
            print(json.dumps({"req": name, "setting": k[1], "score27_of_choice": s27}), flush=True)
    print("@@JSON@@" + json.dumps(rows))


if __name__ == "__main__":
    main()
