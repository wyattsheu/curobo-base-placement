"""Entry point: 3D SLAM map + upstream request → where Kachaka should go, checked that the arm can still grasp there.

  python run_demo.py                                           # test_maps/slam.npz + test_maps/request.json
  python run_demo.py --npz /maps/gs_map_….npz --request request.json
  python run_demo.py --lab-map robot_ws/data/lab/map803_0930 --request request.json     # the lab's exported map

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
  result["arm"]["traj"]                         6-DoF joint trajectory from the stowed arm (cuRobo MotionPlanner)
"""
import argparse
import json
import math
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--npz", default=str(HERE / "test_maps/slam.npz"))
    ap.add_argument("--lab-map", default=None, help="the lab's exported map folder (sem_color_fine.npz + instance table + alignment)")
    ap.add_argument("--request", default=str(HERE / "test_maps/request.json"))
    ap.add_argument("--occupancy", default=None, help="Kachaka's 2D map yaml: park only on its free cells (auto with --lab-map)")
    ap.add_argument("--ignore-classes", nargs="*", default=["floor"], help="classes that are never obstacles")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    req = scene_io.load_request(args.request, cfg)
    if args.lab_map:
        from lab_map import load_lab_map
        data, names, _ = load_lab_map(args.lab_map)
        occ = sorted(Path(args.lab_map).glob("occupancy_*.yaml"))
        args.occupancy = args.occupancy or (str(occ[0]) if occ else None)
    else:
        data, names = args.npz, scene_io.load_class_names(Path(args.npz).with_suffix(".json"))
    g = req["grasp_pose_in_map"]
    scene = scene_io.scene_from_slam(data, req["base_pose_in_map"], g["position"], g["quaternion_wxyz"], cfg,
                                     target=req.get("target"), class_names=names, ignore_classes=args.ignore_classes,
                                     occupancy=args.occupancy)
    from placement import BasePlacementPlanner                                # needs cuRobo + GPU
    res = BasePlacementPlanner(cfg, scene).plan()
    res.update({"n_boxes": len(scene.boxes_base), "boxes_base": scene.boxes_base, "base_pose_in_map": list(scene.base_pose_in_map),
                "grasp_pose_in_map": g, "grasp_pos_base": list(scene.grasp_pos_base), "grasp_quat_base": list(scene.grasp_quat_base),
                "target_points": scene.target_points, "target_source": scene.target_source, "known_source": scene.known_source})
    if res.get("ok"):
        res["at_goal"] = pose_at_goal(cfg, scene.grasp_pos_base, scene.grasp_quat_base, res["chosen_base"])
    print("@@JSON@@" + json.dumps(res, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))
    return res


if __name__ == "__main__":
    main()
