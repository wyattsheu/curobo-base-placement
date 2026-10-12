"""Entry point: 3D SLAM map + upstream request → where Kachaka should go, checked that the arm can still grasp there.

  python run_demo.py                                           # test_maps/slam.npz + test_maps/request.json
  python run_demo.py --npz /maps/gs_map_….npz --request request.json

request.json (written by upstream, see scene_io.load_request):
  {"base_pose_in_map": [x, y, yaw],
   "grasp_pose_in_map": {"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]},
   "target": {"mask_key": "target_mask"}}        # or {"class": "can"}
The class-name table is read from the .json saved next to the .npz (RunSlam convention).
The printed JSON line starts with @@JSON@@; result["chosen_map"] is the (x, y, yaw) goal for Navigate_to_pose.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from robot_model import load_config
import scene_io

HERE = Path(__file__).parent


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--npz", default=str(HERE / "test_maps/slam.npz"))
    ap.add_argument("--request", default=str(HERE / "test_maps/request.json"))
    ap.add_argument("--ignore-classes", nargs="*", default=["floor"], help="classes that are never obstacles")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    req = scene_io.load_request(args.request)
    names_json = Path(args.npz).with_suffix(".json")
    names = json.loads(names_json.read_text(encoding="utf-8")) if names_json.exists() else {}
    g = req["grasp_pose_in_map"]
    scene = scene_io.scene_from_slam(args.npz, req["base_pose_in_map"], g["position"], g["quaternion_wxyz"], cfg,
                                     target=req.get("target"), class_names=names, ignore_classes=args.ignore_classes)
    from placement import BasePlacementPlanner                                # needs cuRobo + GPU
    res = BasePlacementPlanner(cfg, scene).plan()
    res.update({"n_boxes": len(scene.boxes_base), "boxes_base": scene.boxes_base, "base_pose_in_map": list(scene.base_pose_in_map),
                "grasp_pos_base": list(scene.grasp_pos_base), "grasp_quat_base": list(scene.grasp_quat_base),
                "target_points": scene.target_points, "target_source": scene.target_source})
    print("@@JSON@@" + json.dumps(res, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))
    return res


if __name__ == "__main__":
    main()
