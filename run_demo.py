"""Entry point: 3D SLAM map + upstream request → where Kachaka should go, checked that the arm can still grasp there.

  python run_demo.py                                           # test_maps/slam.npz + test_maps/request.json
  python run_demo.py --npz /maps/gs_map_….npz --request request.json
  python run_demo.py --lab-map robot_ws/data/lab/map803_0930 --request request.json     # the lab's exported map
  python run_demo.py --lab-map <map> --serve < requests.txt     # solvers built once; one request per line (path or JSON)
  python run_demo.py --lab-map <map> --request request.json --arm   # separate step, once Kachaka is there: arm trajectory

request.json (written by upstream, see scene_io.load_request):
  {"base_pose_in_map": [x, y, yaw],                                    # Kachaka localisation (kachaka_base_link)
   "grasp_pose_in_map":        {"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]},
   # or, as mm_actions works, the same pose in the Piper base frame of the robot standing at base_pose_in_map:
   "grasp_pose_in_piper_base": {"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]},
   "target": {"instance_id": 13}}       # or {"mask_key": "target_mask"} / {"class": "bottle"}
The class-name table is read from the .json next to the .npz, or from the lab export's instance table.
The printed JSON line starts with @@JSON@@:
  result["chosen_map"]                          (x, y, yaw) goal for Navigate_to_pose (same form as Stage1's goal)
  result["at_goal"]["grasp_pose_in_piper_base"] the decided gripper pose seen from the Piper base once Kachaka
                                                stands at chosen_map → mm_actions move_arm_to_pose
The arm trajectory is NOT part of this answer. Once Kachaka stands at chosen_map and upstream has looked at the target
again, send that new request with --arm (or "arm": true in --serve): the robot is taken to stand where it is now, and
  result["arm"]["traj"]                         6-DoF joint trajectory from the stowed arm (cuRobo MotionPlanner)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from robot_model import load_config, arm_mount
import scene_io

HERE = Path(__file__).parent


def pose_at_goal(cfg, grasp_pos_base, grasp_quat_base, chosen_base):
    """The decided gripper pose in the Kachaka base frame and in the Piper base frame, with Kachaka standing at
    chosen_base (all inputs in the planning base frame)."""
    cx, cy, ct = chosen_base
    c, s = math.cos(-ct), math.sin(-ct)
    dx, dy = grasp_pos_base[0] - cx, grasp_pos_base[1] - cy
    p_k = [c * dx - s * dy, s * dx + c * dy, float(grasp_pos_base[2])]
    q_k = scene_io._qmul([math.cos(-ct / 2), 0.0, 0.0, math.sin(-ct / 2)], grasp_quat_base)
    mxyz, mq = arm_mount(cfg); mq_inv = [mq[0], -mq[1], -mq[2], -mq[3]]
    p_p = scene_io._qrot(mq_inv, np.subtract(p_k, mxyz).tolist())
    q_p = scene_io._qmul(mq_inv, q_k)
    f = lambda v: [float(x) for x in v]
    return {"grasp_pose_in_kachaka_base": {"position": f(p_k), "quaternion_wxyz": f(q_k)},
            "grasp_pose_in_piper_base": {"position": f(p_p), "quaternion_wxyz": f(q_p)}}


def load_map(args, cache):
    """(data, class names, occupancy yaml) of the map a request uses; maps are read once per process."""
    key = ("lab", args.lab_map) if args.lab_map else ("npz", args.npz)
    if key not in cache:
        if args.lab_map:
            from lab_map import load_lab_map
            data, names, _ = load_lab_map(args.lab_map)
            occ = sorted(Path(args.lab_map).glob("occupancy_*.yaml"))
            cache[key] = (data, names, str(occ[0]) if occ else None)
        else:
            with np.load(args.npz) as z:                                        # read once, kept in memory
                data = {k: z[k] for k in z.files}
            cache[key] = (data, scene_io.load_class_names(Path(args.npz).with_suffix(".json")), None)
    return cache[key]


def solve(planner_or_none, cfg, args, req, cache, arm=False):
    """One request → result dict. With a long-running planner the solvers are reused; otherwise built for this scene.
    arm: Kachaka already stands where it should — plan the arm trajectory instead of a base pose."""
    t0 = time.perf_counter()
    data, names, occ = load_map(args, cache)
    g = req["grasp_pose_in_map"]
    scene = scene_io.scene_from_slam(data, req["base_pose_in_map"], g["position"], g["quaternion_wxyz"], cfg,
                                     target=req.get("target"), class_names=names, ignore_classes=args.ignore_classes,
                                     occupancy=args.occupancy or occ)
    t_scene = time.perf_counter() - t0
    from placement import BasePlacementPlanner                                # needs cuRobo + GPU
    if planner_or_none is None:
        planner = BasePlacementPlanner(cfg, scene)
    else:
        planner = planner_or_none; planner.set_scene(scene)
    if arm:                                   # separate step: the robot already stands there, only the arm moves
        traj = planner.plan_arm()
        return {"ok": traj is not None, "reason": "ok" if traj is not None else "arm_plan_failed", "arm": traj,
                "timing_s": {"scene_from_slam_s": t_scene, "arm_plan_s": planner.T.get("arm_plan_s", 0.0),
                             "request_s": time.perf_counter() - t0}}
    res = planner.plan()
    res["timing_s"]["scene_from_slam_s"] = t_scene
    res.update({"n_boxes": len(scene.boxes_base), "boxes_base": scene.boxes_base, "base_pose_in_map": list(scene.base_pose_in_map),
                "grasp_pose_in_map": g, "grasp_pos_base": list(scene.grasp_pos_base), "grasp_quat_base": list(scene.grasp_quat_base),
                "target_points": scene.target_points, "target_source": scene.target_source, "known_source": scene.known_source})
    if res.get("ok"):
        res["at_goal"] = pose_at_goal(cfg, scene.grasp_pos_base, scene.grasp_quat_base, res["chosen_base"])
    res["timing_s"]["request_s"] = time.perf_counter() - t0
    return res


def emit(res):
    print("@@JSON@@" + json.dumps(res, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--npz", default=str(HERE / "test_maps/slam.npz"))
    ap.add_argument("--lab-map", default=None, help="the lab's exported map folder (sem_color_fine.npz + instance table + alignment)")
    ap.add_argument("--request", default=str(HERE / "test_maps/request.json"))
    ap.add_argument("--occupancy", default=None, help="Kachaka's 2D map yaml: park only on its free cells (auto with --lab-map)")
    ap.add_argument("--ignore-classes", nargs="*", default=["floor"], help="classes that are never obstacles")
    ap.add_argument("--arm", action="store_true", help="separate step once Kachaka has arrived: plan the arm trajectory "
                    "from where it stands (no base search)")
    ap.add_argument("--serve", action="store_true",
                    help="build the solvers once, then answer one request per stdin line: a request.json path or the request "
                         "itself as JSON; optional keys \"lab_map\" / \"npz\" / \"occupancy\" switch maps (each map is read once)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cache = {}
    if not args.serve:
        res = solve(None, cfg, args, scene_io.load_request(args.request, cfg), cache, arm=args.arm)
        emit(res)
        return res
    from placement import BasePlacementPlanner
    planner = BasePlacementPlanner(cfg)
    print(f"ready (setup {planner.T['setup_s']:.1f}s)", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line) if line.startswith("{") else {"request": line}
            req = msg.get("request", msg)
            req = scene_io.load_request(req, cfg) if isinstance(req, str) else scene_io.normalize_request(req, cfg)
            a = argparse.Namespace(**{**vars(args), **{k: msg[k] for k in ("lab_map", "npz", "occupancy") if k in msg}})
            emit(solve(planner, cfg, a, req, cache, arm=bool(msg.get("arm"))))
        except Exception as e:                                                  # keep serving
            emit({"ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
