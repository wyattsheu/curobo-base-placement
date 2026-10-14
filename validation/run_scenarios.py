"""Check that each `reason` in plan()'s result comes out where it should (needs cuRobo + GPU).

  python validation/run_scenarios.py                    # all scenarios
  python validation/run_scenarios.py no_drivable_area   # only the named ones

Each scenario is the test scene (test_maps/) with one thing changed, in memory only: nothing outside this folder is
modified, and nothing outside this folder imports it. Coordinates are in the test scene's base frame (= test_scene.yaml).
"""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot_model import load_config  # noqa: E402
import scene_io  # noqa: E402


# ---------------------------------------------------------------- what a scenario may change
def cfg_set(**kv):
    """Config overrides, keys as 'section.key'."""
    def apply(cfg, scene):
        for path, v in kv.items():
            *head, last = path.split(".")
            d = cfg
            for k in head:
                d = d[k]
            d[last] = v
    return apply


def grasp_at(pos):
    def apply(cfg, scene):
        scene.grasp_pos_base = list(pos)
    return apply


def no_drive_near_grasp(radius):
    """Cells within `radius` of the gripper position become not drivable (as if Kachaka's 2D map said so)."""
    def apply(cfg, scene):
        k = scene.known
        H, W = k.known.shape
        jj, ii = np.mgrid[0:H, 0:W]
        xy = scene_io.map_to_base_xy(np.c_[ii.ravel(), jj.ravel()] * k.resolution + k.origin_xy, k.base_pose_in_map)
        near = np.hypot(*(xy - scene.grasp_pos_base[:2]).T) < radius
        k.known = k.known & ~near.reshape(H, W)
    return apply


# ---------------------------------------------------------------- scenarios: (name, expected reason, change)
SCENARIOS = [
    ("baseline",          "ok",              None),
    ("loose_threshold",   "ok",              cfg_set(**{"robustness.threshold": 0.5})),
    ("unreachable",       "no_ik_solution",  grasp_at([0.92, 0.0, 2.0])),
    ("gripper_in_cabinet", "no_ik_solution", grasp_at([0.85, 0.75, 0.45])),
    ("cone_closed",       "outside_cone",    cfg_set(**{"robot.bearing_limit_rad": 0.0})),
    ("no_drivable_area",  "cannot_park",     no_drive_near_grasp(1.2)),
    ("sloppy_navigation", "not_robust",      cfg_set(**{"robustness.perturb_xy_m": 0.15, "robustness.perturb_yaw_deg": 15.0})),
    ("bad_stow",          "arm_plan_failed", cfg_set(**{"robot.stow_arm": [0.0] * 6})),
]


def load_test_scene():
    cfg = load_config(ROOT / "config.yaml")
    req = scene_io.load_request(ROOT / "test_maps/request.json", cfg)
    g = req["grasp_pose_in_map"]
    scene = scene_io.scene_from_slam(str(ROOT / "test_maps/slam.npz"), req["base_pose_in_map"], g["position"],
                                     g["quaternion_wxyz"], cfg, target=req.get("target"),
                                     class_names=scene_io.load_class_names(ROOT / "test_maps/slam.json"))
    return cfg, scene


def main(names):
    from placement import BasePlacementPlanner
    cfg0, scene0 = load_test_scene()
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    failed = 0
    for name, want, change in SCENARIOS:
        if names and name not in names:
            continue
        cfg, scene = copy.deepcopy(cfg0), copy.deepcopy(scene0)
        if change:
            change(cfg, scene)
        r = BasePlacementPlanner(cfg, scene).plan()
        ok = r["reason"] == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:18s} reason={r['reason']:16s} expected={want:16s} "
              f"ik={r['n_success']} cone={r['n_cone']} park={sum(r['parked_ok'])} scored={r['n_checked']} "
              f"robustness={fmt(r['robustness'])} max={fmt(r['max_robustness'])} plans={len(r['tried_plans'])}",
              flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(set(sys.argv[1:])))
