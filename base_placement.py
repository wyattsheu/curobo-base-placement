"""The one entry point for the robot system: where should Kachaka stand to do one of these grasps?
Called only when mm_actions cannot plan any of the grasps from where the robot stands now.

In Python (inside the mm-gpu image):

    from base_placement import BasePlacement
    bp = BasePlacement("/maps/map803_0930")          # once, when the system starts (~15 s on the Thor)
    r = bp.find_standpoint(robot_pose, grasps, object_pts, scene_pts)
    if r["ok"]:
        kachaka.move_to_pose(*r["pose"])              # (x, y, yaw), map frame

Inputs, exactly as mm_actions already holds them:
    robot_pose  (x, y, yaw): Kachaka now, map frame (same as KachakaApiClient.get_robot_pose(), m / rad)
    grasps      list of 4x4 GraspGenX grasps (grasp_frame) in the Piper base frame, best first; one 4x4 is fine too
    object_pts  (optional) (N, 3) the target's points from the wrist camera, Piper base frame (build_clouds().object_pts)
    scene_pts   (optional) (M, 3) the rest of the wrist view, Piper base frame (build_clouds().scene_pts)
    map_from_arm (optional) 4x4 Piper base → map from Grasp Lab Global; without it the arm mount in config.yaml is used
Without object_pts the target is taken as an upright cylinder around the gripper (fine for a bottle or a can).

Result: {"ok", "pose": [x, y, yaw] or None, "grasp_index", "reason", "robustness", "fallback", "time_s", "tried"}.
Grasps are tried in the given order; the first one with a station that survives every navigation error (e.g. 27/27) wins.
Otherwise the most robust fallback station is returned (fallback = true), or ok = false with the reason of the first grasp.
"""
import math
import time
from pathlib import Path

import numpy as np

from robot_model import load_config, arm_mount
from lab_map import load_lab_map
import grasp_lab
import scene_io

HERE = Path(__file__).parent


def _rot_wxyz(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class BasePlacement:
    def __init__(self, map_dir, config=HERE / "config.yaml", max_grasps=10):
        """Everything slow happens here, once: call it when the system starts (Thor: ~14 s solvers, map 3.8 s the very
        first time it is ever seen and ~7 ms afterwards from .map_cache/, box workers ~1 s). Use one instance per
        process, and call it from one thread. Times are kept in self.startup_s."""
        from placement import BasePlacementPlanner                             # needs cuRobo + GPU
        t = time.perf_counter()
        self.cfg = load_config(str(config))
        self.max_grasps = max_grasps
        self.planner = BasePlacementPlanner(self.cfg)                          # all cuRobo solvers, warmed up
        self.startup_s = {"solvers": time.perf_counter() - t}
        self.set_map(map_dir)
        t = time.perf_counter()
        scene_io.warm_box_pool(self.cfg["map"].get("box_workers", 0))           # CPU workers for the obstacle boxes
        self.startup_s["box_workers"] = time.perf_counter() - t

    def set_map(self, map_dir):
        """Load (or switch to) a map: the lab's exported map folder with its occupancy_*.yaml. Cached on disk, so a
        map seen before loads in milliseconds."""
        t = time.perf_counter()
        self.map, self.names, _ = load_lab_map(str(map_dir))
        occ = sorted(Path(map_dir).glob("occupancy_*.yaml"))
        self.occupancy = str(occ[0]) if occ else None
        self.startup_s["map"] = time.perf_counter() - t

    def map_from_arm(self, robot_pose):
        """Piper base → map for Kachaka standing at robot_pose (through the arm mount in config.yaml)."""
        x, y, yaw = robot_pose
        B = np.eye(4); B[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]; B[:2, 3] = (x, y)
        xyz, q = arm_mount(self.cfg)
        A = np.eye(4); A[:3, :3] = _rot_wxyz(q); A[:3, 3] = xyz
        return B @ A

    def find_standpoint(self, robot_pose, grasps, object_pts=None, scene_pts=None, map_from_arm=None, viz_dir=None):
        """viz_dir: also save one picture per grasp tried (top + side view, why it was accepted or rejected); ~1 s each."""
        t0 = time.perf_counter()
        robot_pose = [float(v) for v in robot_pose]
        G = np.asarray(grasps, float).reshape(-1, 4, 4)[:self.max_grasps]
        if map_from_arm is None:
            M = self.map_from_arm(robot_pose)
        else:                                                                    # Grasp Lab Global: z not floor-subtracted
            M = np.array(map_from_arm, float); M[2, 3] -= float(self.map["floor_height_m"])
        to_map = lambda p: np.asarray(p, float).reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        world, target = self.map, None
        if object_pts is not None or scene_pts is not None:                     # add what the wrist camera saw
            cap = {"scene_map": to_map(scene_pts if scene_pts is not None else np.zeros((0, 3))),
                   "object_map": to_map(object_pts if object_pts is not None else np.zeros((0, 3)))}
            world = grasp_lab.fuse(self.cfg, self.map, cap, self.names)
            if len(cap["object_map"]):
                target = {"mask_key": "target_mask"}

        tried = []
        for i, T in enumerate(G):
            P = M @ grasp_lab.pinch_pose(T, self.cfg["robot"]["tool_offset_m"])
            scene = scene_io.scene_from_slam(world, robot_pose, P[:3, 3].tolist(), grasp_lab._quat_wxyz(P[:3, :3]),
                                             self.cfg, target=target, class_names=self.names, occupancy=self.occupancy)
            self.planner.set_scene(scene)
            r = self.planner.plan()
            if viz_dir is not None:
                import viz
                K = r.get("n_perturb", 27); rb = f", {round(r['robustness'] * K)}/{K}" if r["ok"] else ""
                viz.draw(Path(viz_dir) / f"grasp_{i:02d}_{r['reason']}.png", self.cfg, world, scene, r, P, self.occupancy,
                         title=f"grasp #{i}: {r['reason']}{rb}" + (" (fallback)" if r.get("below_threshold") else ""))
            tried.append({"ok": bool(r["ok"]), "reason": r["reason"], "robustness": float(r.get("robustness") or 0.0),
                          "fallback": bool(r.get("below_threshold")),
                          "pose": [float(v) for v in r["chosen_map"]] if r["ok"] else None})
            if r["ok"] and not r.get("below_threshold"):
                break                                                            # passes every error: good enough, stop

        ok = [i for i, t in enumerate(tried) if t["ok"]]
        best = min(ok, key=lambda i: (tried[i]["fallback"], -tried[i]["robustness"], i)) if ok else None
        b = tried[best] if best is not None else (tried[0] if tried else {"reason": "no_grasps", "robustness": 0.0})
        return {"ok": best is not None, "pose": b.get("pose"), "grasp_index": best, "reason": b["reason"],
                "robustness": b["robustness"], "fallback": bool(b.get("fallback")),
                "time_s": round(time.perf_counter() - t0, 3), "tried": tried}

