"""Make stand-in inputs in the formats this module will receive (run locally: python make_test_maps.py):

  test_maps/slam.npz     lab RunSlam output (SLAM_DOCKER_HANDOFF.md §4): means3D (N, 3) float32, semantic_ids (N,) uint8,
                         plus target_mask (N,) bool = the target object's region as labelled by the SLAM
  test_maps/slam.json    class names {id: name}
  test_maps/request.json what upstream hands over: Kachaka localisation, the decided gripper pose, which points are the target

Scene (test_scene.yaml, relative to the robot): table ahead with a can on it, a board in front of the can, cabinet on the
left, walls behind and on the right. Upstream decided a side grasp from the robot's right (gripper approaching along +y,
horizontal, fingers pointing down); from where the robot stands the can is out of reach.
The robot is placed at map pose (2.0, 1.0, 30°) so the map → base transform is actually exercised.
"""
import json
import math
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
OUT = HERE / "test_maps"; OUT.mkdir(exist_ok=True)
for old in ("map.yaml", "map.pgm", "camera_boxes.yaml", "robot_state.json"):        # formats no longer used
    (OUT / old).unlink(missing_ok=True)
S = yaml.safe_load((HERE / "test_scene.yaml").read_text(encoding="utf-8"))
BASE_MAP = (2.0, 1.0, math.radians(30.0))
CLASSES = {0: "floor", 1: "table", 2: "board", 3: "cabinet", 4: "wall", 5: "can"}
CLS = {"table": 1, "blocker": 2, "cabinet": 3, "wall_b": 4, "wall_r": 4}


def to_map(p):
    bx, by, bt = BASE_MAP; c, s = math.cos(bt), math.sin(bt)
    return np.c_[bx + c * p[:, 0] - s * p[:, 1], by + s * p[:, 0] + c * p[:, 1], p[:, 2]]


boxes = [dict(b, yaw=BASE_MAP[2]) for b in S["boxes"] if b["name"] != "floor"]
for b in boxes:
    b["center"] = to_map(np.array([b["center"]]))[0].tolist()
can = to_map(np.array([S["target_in_map"]]))[0]

rng = np.random.default_rng(0)
P, L = [], []
for b in boxes:                                            # points on every box surface, 2 cm apart
    d = np.asarray(b["dims"]); n = np.maximum((d / 0.02).astype(int), 2)
    g = [np.linspace(-d[i] / 2, d[i] / 2, n[i]) for i in range(3)]
    for ax in range(3):
        for sgn in (-1, 1):
            o = [a for a in range(3) if a != ax]
            A, B = np.meshgrid(g[o[0]], g[o[1]])
            loc = np.zeros((A.size, 3)); loc[:, o[0]] = A.ravel(); loc[:, o[1]] = B.ravel(); loc[:, ax] = sgn * d[ax] / 2
            c, s = math.cos(b["yaw"]), math.sin(b["yaw"])
            P.append(np.c_[b["center"][0] + c * loc[:, 0] - s * loc[:, 1], b["center"][1] + s * loc[:, 0] + c * loc[:, 1], b["center"][2] + loc[:, 2]])
            L.append(np.full(A.size, CLS[b["name"]]))
th = np.linspace(0, 2 * np.pi, 40); zz = np.linspace(-0.06, 0.06, 7)    # the can (target), r = 3.3 cm, 12 cm tall
T_, Z_ = np.meshgrid(th, zz)
P.append(np.c_[can[0] + 0.033 * np.cos(T_.ravel()), can[1] + 0.033 * np.sin(T_.ravel()), can[2] + Z_.ravel()]); L.append(np.full(T_.size, 5))
gx, gy = np.meshgrid(np.arange(-1.0, 1.40, 0.04), np.arange(-1.30, 1.25, 0.04))   # floor the camera saw (room, robot frame)
fl = to_map(np.c_[gx.ravel(), gy.ravel(), np.zeros(gx.size)])
inside_any = np.zeros(len(fl), bool)
for b in boxes:
    c = np.asarray(b["center"]); d = np.asarray(b["dims"]) / 2; cy, sy = math.cos(-b["yaw"]), math.sin(-b["yaw"])
    q = fl - c; inside_any |= (np.abs(cy * q[:, 0] - sy * q[:, 1]) <= d[0]) & (np.abs(sy * q[:, 0] + cy * q[:, 1]) <= d[1])
P.append(fl[~inside_any]); L.append(np.zeros((~inside_any).sum(), int))
P = np.vstack(P) + rng.normal(0, 0.003, (sum(len(p) for p in P), 3))
L = np.concatenate(L).astype(np.uint8)
np.savez_compressed(OUT / "slam.npz", means3D=P.astype(np.float32), semantic_ids=L, target_mask=(L == 5))
json.dump(CLASSES, open(OUT / "slam.json", "w"), ensure_ascii=False, indent=1)

# decided gripper pose (upstream): at the can centre; tool z (approach) = +y of the robot's start frame, tool x = down
x = np.array([0.0, 0.0, -1.0]); z = np.array([0.0, 1.0, 0.0]); y = np.cross(z, x)
R = np.stack([x, y, z], 1)
w = math.sqrt(1 + R.trace()) / 2
q_robot = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
h = BASE_MAP[2] / 2                                         # into the map frame: Rz(30°) ⊗ q
w1, z1 = math.cos(h), math.sin(h); w2, x2, y2, z2 = q_robot
q_map = [w1 * w2 - z1 * z2, w1 * x2 - z1 * y2, w1 * y2 + z1 * x2, w1 * z2 + z1 * w2]
req = {"base_pose_in_map": list(BASE_MAP),
       "grasp_pose_in_map": {"position": can.tolist(), "quaternion_wxyz": q_map},
       "target": {"mask_key": "target_mask"}}
json.dump(req, open(OUT / "request.json", "w"), indent=1)
print("slam.npz points", len(P), "| target points", int((L == 5).sum()), "| grasp position (map)", np.round(can, 3),
      "| quaternion (map)", np.round(q_map, 4))
