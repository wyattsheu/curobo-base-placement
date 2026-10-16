# Usage

Everything you need to call the planner from code or from the command line. Times are medians measured on the Jetson
Thor (120 W mode, GPU shared with the lab's Qwen / Molmo containers). For running it as a service on the robot, see
[deployment.md](deployment.md).

## The whole flow

Three things cost time once; everything per request is small. **Build the planner and load the map at startup, never
per request.**

```python
from robot_model import load_config
from placement import BasePlacementPlanner
from lab_map import load_lab_map
import scene_io, grasp_lab

cfg = load_config("config.yaml")

# ---- 1. at startup, once -----------------------------------------------------------------------------------------
planner = BasePlacementPlanner(cfg)                    # ~14 s: every cuRobo solver built and warmed up
slam, names, _ = load_lab_map("/maps/map803_0930")     # ~3.8 s the first time a map is ever seen, then ~7 ms (.map_cache/)
occupancy = "/maps/map803_0930/occupancy_map803_0930.yaml"   # Kachaka's 2D map: where the base may park

# ---- 2. once per wrist capture (optional: only if you have a Grasp Lab capture, see grasp_lab.md) ----------------
cap = grasp_lab.load_capture("/grasp_logs/20261008_045944_bottle", float(slam["floor_height_m"]), "/maps/map803_0930")
world = grasp_lab.fuse(cfg, slam, cap, names)          # ~0.19 s: SLAM map + what the wrist camera saw
base_pose = grasp_lab.kachaka_pose(cfg, cap["map_from_arm"])   # where Kachaka stands now (x, y, yaw), map frame
#     without a capture: world = slam, base_pose = Kachaka's localisation, target={"instance_id": ...} or {"class": ...}

# ---- 3. once per candidate grasp ----------------------------------------------------------------------------------
for grasp in grasps:                                   # each {"position": [x, y, z], "quaternion_wxyz": [...]}, map frame
    scene = scene_io.scene_from_slam(world, base_pose, grasp["position"], grasp["quaternion_wxyz"], cfg,
                                     target={"mask_key": "target_mask"}, occupancy=occupancy)    # ~0.18 s
    planner.set_scene(scene)
    result = planner.plan()                            # 0.3–0.8 s with a 27/27 station; ~1.5 s if impossible
    if result["ok"]:
        x, y, yaw = result["chosen_map"]               # → Navigate_to_pose
    else:
        print("rejected:", result["reason"])           # see "Outputs" below

# ---- 4. after Kachaka has arrived and upstream has looked at the target again (separate step) --------------------
planner.set_scene(scene_io.scene_from_slam(world, arrived_pose, new_grasp_pos, new_grasp_quat, cfg, ...))
arm = planner.plan_arm()                               # ~0.3 s: {"traj": N × 6 joint angles, "dt": s} or None
```

Scripts that call these functions need `if __name__ == "__main__":` (box fitting uses a pool of CPU processes; without
the guard it falls back to one core).

## Many candidate grasps

[`demo_multi_grasp.py`](../demo_multi_grasp.py) runs the whole flow for one capture with N grasps (the executed one, the
best GraspGen candidates, and one deliberately impossible grasp 2 m up in the air), rejects the impossible ones with a
reason, and picks one:

```bash
python demo_multi_grasp.py --lab-map <map dir> --run <Grasp Lab run dir> [--top 8] [--first] [--drop-stale]
```

Output on the Thor (map803_0930, run 20261008_045944_bottle):

```
executed               drive to (2.36, 0.32, 68°), 0.02 m away, robustness 27/27    scene 0.62 s + plan 0.36 s
graspgen0              REJECTED: no_ik_solution                                     scene 0.17 s + plan 1.51 s
graspgen2              drive to (2.93, 0.28, 137°), 0.54 m away, robustness 27/27   scene 0.18 s + plan 0.80 s
graspgen3              drive to (2.56, 0.17, 77°), 0.22 m away, robustness 15/27 (fallback) scene 0.18 s + plan 7.23 s
graspgen5              REJECTED: not_robust                                         scene 0.19 s + plan 6.65 s
impossible (2 m up)    REJECTED: no_ik_solution                                     scene 0.16 s + plan 1.48 s
...
picked: executed → drive to (2.364, 0.317, 1.18)
startup: planner 14.2 s, SLAM map 0.007 s (once) | capture merge 0.19 s | 10 grasps: 29.9 s in all, median 1.67 s each
```

How to pick (`pick()` in the demo; a simple rule, still to be agreed with upstream): a station that survives every
navigation error first, then the most robust, then the shortest drive, then upstream's own grasp confidence.

Trying all N is slow when some grasps only have weak stations (5–7 s each). In practice, try the grasps in upstream's
confidence order and **stop at the first 27/27** (`--first`): here that is the first grasp, about 1 s.

## Command line

```bash
python run_demo.py                                                   # test_maps/slam.npz + test_maps/request.json
python run_demo.py --npz map.npz --request request.json              # one request, solvers built for it
python run_demo.py --lab-map /maps/map803_0930 --request request.json
python run_demo.py --lab-map /maps/map803_0930 --serve < requests.txt   # long-running, see deployment.md
python run_demo.py --lab-map /maps/map803_0930 --request arrived.json --arm   # arm trajectory only (after arriving)
```

Every request prints one line `@@JSON@@{...}`.

## Inputs

### The SLAM point cloud: `.npz`

| Field | Shape / type | Required | Use |
|---|---|---|---|
| `means3D` | (N, 3) float32, metres | **yes** | point positions in the **map frame** (the frame Kachaka localises in), z up, floor at z ≈ 0 |
| `semantic_ids` | (N,) uint8 | no | class id per point; objects are first separated by class |
| target mask (any name, e.g. `target_mask`) | (N,) bool | recommended | the **target object's points**: they are what is grasped, never an obstacle |
| `logit_opacities` | (N, 1) | no | 3D Gaussian opacity; only points with > 0 are used |
| `<same name>.json` | `{"0": "floor", "1": "table", …}` | no | class names, so a request may name a class |

No obstacle boxes or grids are needed: obstacles are computed from the points. Every point above the floor, within
1.8 m of the gripper and not part of the target is an obstacle. Other formats only need converting:

```python
np.savez("map.npz", means3D=xyz.astype(np.float32))                          # minimum
np.savez("map.npz", means3D=xyz.astype(np.float32), target_mask=is_target)   # recommended
```

### The request

```json
{"base_pose_in_map": [2.0, 1.0, 0.5236],
 "grasp_pose_in_map": {"position": [2.797, 1.46, 0.46], "quaternion_wxyz": [0.354, -0.612, 0.354, 0.612]},
 "target": {"mask_key": "target_mask"}}
```

| Field | Meaning |
|---|---|
| `base_pose_in_map` | where Kachaka is **now**: (x, y, yaw), map frame, yaw in radians |
| `grasp_pose_in_map.position` | the decided gripper position (`pinch_center`, between the two fingers), map frame, metres |
| `grasp_pose_in_map.quaternion_wxyz` | the decided gripper orientation, w, x, y, z; `pinch_center`'s +z is the approach direction |
| `grasp_pose_in_piper_base` | alternative to `grasp_pose_in_map`, as mm_actions uses it: the same pose in the Piper base frame of the robot standing at `base_pose_in_map` |
| `target` (optional) | which points are the target, best first: `{"mask_key": "target_mask"}` (a bool field of the npz); `{"instance_id": 13}` (lab maps); `{"class": "can"}` (that class within 0.2 m of the gripper; farther ones stay obstacles); omitted → an upright cylinder of radius 7 cm, ±10 cm around the gripper (fine for a small upright can or bottle) |

Why the target matters: it is in the point cloud and the gripper pose is inside it. Not excluded, it becomes an
obstacle and the gripper "collides" at the goal, so IK never succeeds.

### The lab's data

| What the lab has | How to give it | Notes |
|---|---|---|
| exported 3D semantic map folder (`robot_ws/data/lab/<site>/`) | `--lab-map <dir>` / `lab_map.load_lab_map(dir)` | reads `*_sem_color_fine.npz`, `*_instance_labels.npz`, `*_instance_semantic_table_fine.json`, `alignment_*.yaml`; applies that alignment (the object_query formula) and moves the floor to z = 0 |
| Kachaka's 2D map `occupancy_<site>.yaml/png` | read automatically with `--lab-map`, or `--occupancy` | the base may only park on its free cells; Stage1 uses the same map |
| object_query instance ids | `"target": {"instance_id": 13}` | |
| Grasp Lab Global captures (`main_ws/grasp_logs/<run>/`) | `grasp_lab.py` | [grasp_lab.md](grasp_lab.md) |

## Outputs

`plan()` returns a dict; `run_demo.py` prints it as `@@JSON@@{...}`. Check `ok` and `reason` first.

| Field | Meaning |
|---|---|
| **`ok`** | true = a station was found |
| **`reason`** | `ok`, or the first step that ran out of candidates (below) |
| **`chosen_map`** | **(x, y, yaw), map frame → `Navigate_to_pose`** (only when `ok`) |
| `chosen_base` | the same station relative to where Kachaka is now |
| **`at_goal.grasp_pose_in_piper_base`** | the gripper pose in the Piper base frame once Kachaka stands at `chosen_map` → mm_actions `move_arm_to_pose` (added by `run_demo.py`) |
| `robustness` | fraction of the 27 navigation errors (±5 cm, ±5°) from which the arm still reaches the pose |
| `below_threshold` | true = no station reached `threshold`; this is the fallback |
| `max_robustness` | best robustness among the scored stations (how far off a `not_robust` was) |
| `timing_s`, `memory_mb` | time per step (s), GPU memory (MB) |
| `n_boxes`, `boxes_base`, `target_points`, `target_source`, `known_source` | the obstacles built, how the target was found, what defined the drivable area |
| `n_success`, `n_cone`, `parked_ok`, `n_checked`, `q_success`, `uniq`, `rob` | candidates left after each step and their scores (debugging, plots) |

**Threshold and fallback** (`robustness.threshold`, `min_accept`):
1. Stations are scored nearest first (the current pose first of all); the first with robustness ≥ `threshold`
   (1.0 = all 27) is taken.
2. If none reaches it, the most robust scored station is taken (nearest on ties) as long as it reaches `min_accept`
   (0.5); `below_threshold = true`.
3. Otherwise `ok = false`, `reason = not_robust`. A station that only survives one or two errors is never returned.

**`reason`**, the first step that removed every candidate:

| reason | Meaning | What upstream can do |
|---|---|---|
| `ok` | a station was found | drive to `chosen_map` |
| `no_ik_solution` | no 9-DoF IK solution in the search window: out of reach, or the gripper pose itself collides | another grasp |
| `outside_cone` | solutions exist, but none keeps the target within the arm's ±30° safety cone | another grasp, or widen the cone |
| `cannot_park` | solutions exist, but the base can never park there (blocked, or off Kachaka's free cells) | check the map, or another grasp |
| `not_robust` | parkable, but no station reaches `min_accept`; see `max_robustness` | lower `min_accept` at your own risk, or another grasp |
| `arm_plan_failed` | only from `plan_arm()`: no arm trajectory from the stowed arm | check `robot.stow_arm`, retry |

`no_ik_solution` means sampling found nothing (128 seeds × up to 4 rounds), not a proof. A colliding gripper pose is
also reported this way for now.

**The arm trajectory is a separate step.** `plan()` only answers where to drive. Once Kachaka has arrived and upstream
has looked at the target again, send a new request with the pose actually reached: `set_scene(new scene)` then
`plan_arm()` (base locked, 6-DoF MotionPlanner from the stowed arm). It returns `{"traj": N × 6 (joint1–6, rad), "dt"}`
or `None`. The MotionPlanner is built on the first call (~2–3 s) unless `profile.build_arm_planner: true`.

## Configuration (`config.yaml`)

Everything measurable lives in `config.yaml`; when you measure the real value, change only this file.

| What | Key | Current value |
|---|---|---|
| arm mount on Kachaka | `robot.arm_mount_xyz`, `arm_mount_rpy_deg` | (0.08, 0, 0.525) m, lab URDF |
| gripper frame of the requested pose | `robot.tool_frame`, `tool_offset_m` | `pinch_center`, 0.12 m in front of `gripper_base` |
| Kachaka body | `robot.kachaka.size_m`, `extra_spheres` | 0.387 × 0.240 × 0.095 m; camera pole not yet added |
| arm pose while driving / start of `plan_arm` | `robot.stow_arm` | mm_actions' carry pose |
| safety cone | `robot.bearing_limit_rad` | ±30° |
| navigation error | `robustness.perturb_xy_m`, `perturb_yaw_deg` | ±5 cm, ±5° (**assumed — measure Kachaka**) |
| acceptance | `robustness.threshold`, `min_accept` | 1.0, 0.5 |
| search window | `search.window_x`, `window_y` | ±1.5 m around the current pose |
| point cloud → obstacles | `map.*` | 4 cm voxels for boxes; ESDF ±0.25 m, 1 cm |
| compute budget | `profile.*` | 128 whole-body seeds, 8 scoring seeds |

Options measured but off by default: `robustness.perturb_points: 9`, `profile.bound_scoring`, `map.crop_to_window`
(see [benchmarks.md](benchmarks.md)).
