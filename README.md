# curobo-base-placement: where should Kachaka stand to grasp?

Upstream has already decided **what to grasp** and **the gripper's final pose** (position + orientation). This project
answers one question:

> Given the **3D SLAM point cloud** and that **gripper pose**, where should the Kachaka base drive to (the goal for
> `Navigate_to_pose`), so that from there the arm can still reach **the same gripper pose** without hitting anything —
> even if Kachaka stops a few centimetres or degrees off?

It does **not** choose grasps, change the gripper pose, plan the navigation path, or build/align maps.

Robot: Kachaka mobile base + AgileX Piper 6-DoF arm (Robot129). Planner: cuRobo v2, 9-DoF whole-body IK
(base x, y, yaw + 6 arm joints), on an NVIDIA Jetson Thor or any CUDA GPU.

| Bottle out of reach: drive there | Already beside it: small correction |
|---|---|
| ![far](docs/media/far.gif) | ![near](docs/media/near.gif) |
| [far.mp4](docs/media/far.mp4) — starts 1.0–1.4 m away, drives 0.34 m | [near.mp4](docs/media/near.mp4) — 21/27 where it stands, moves 0.10 m to 27/27 |
| **A board in front of the bottle** | **A box on the floor in front of Kachaka** |
| ![board](docs/media/board.gif) | ![box](docs/media/box.gif) |
| [board.mp4](docs/media/board.mp4) — the frontal grasp is rejected; a side grasp works after driving 0.51 m | [box.mp4](docs/media/box.mp4) — 9/27 where it stands, moves 0.17 m, the arm reaches over the box |

All four use the lab's map803_0930 SLAM map with a real wrist capture (Grasp Lab Global, see [§8](#8-the-teammates-aligned-wrist-scene-grasp-lab-global));
the board and the box are added as point clouds. Stations come from `plan()`, the arm motion from `plan_arm()`.

---

## Contents

1. [How to call it](#1-how-to-call-it)
2. [Deploying on the robot](#2-deploying-on-the-robot)
3. [Inputs](#3-inputs)
4. [Outputs](#4-outputs)
5. [How it works](#5-how-it-works)
6. [Installation](#6-installation)
7. [Configuration](#7-configuration-configyaml)
8. [The teammate's aligned wrist scene (Grasp Lab Global)](#8-the-teammates-aligned-wrist-scene-grasp-lab-global)
9. [Validation and measurements](#9-validation-and-measurements)
10. [Files](#10-files)
11. [Known limitations and open items](#11-known-limitations-and-open-items)

---

## 1. How to call it

Three things cost time once; everything per request is small. **Build the planner and load the map at startup, never
per request.** Times are medians measured on the Jetson Thor (120 W mode, GPU shared with the lab's Qwen / Molmo
containers).

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

# ---- 2. once per wrist capture ----------------------------------------------------------------------------------
cap = grasp_lab.load_capture("/grasp_logs/20261008_045944_bottle", float(slam["floor_height_m"]), "/maps/map803_0930")
world = grasp_lab.fuse(cfg, slam, cap, names)          # ~0.19 s: SLAM map + what the wrist camera saw
base_pose = grasp_lab.kachaka_pose(cfg, cap["map_from_arm"])   # where Kachaka stands now (x, y, yaw), map frame

# ---- 3. once per candidate grasp ----------------------------------------------------------------------------------
for grasp in grasps:                                   # each {"position": [x, y, z], "quaternion_wxyz": [...]}, map frame
    scene = scene_io.scene_from_slam(world, base_pose, grasp["position"], grasp["quaternion_wxyz"], cfg,
                                     target={"mask_key": "target_mask"}, occupancy=occupancy)    # ~0.18 s
    planner.set_scene(scene)
    result = planner.plan()                            # 0.3–0.8 s with a 27/27 station; ~1.5 s if impossible
    if result["ok"]:
        x, y, yaw = result["chosen_map"]               # → Navigate_to_pose
    else:
        print("rejected:", result["reason"])           # no_ik_solution / outside_cone / cannot_park / not_robust

# ---- 4. after Kachaka has arrived and upstream has looked at the target again (separate step) --------------------
planner.set_scene(scene_io.scene_from_slam(world, arrived_pose, new_grasp_pos, new_grasp_quat, cfg, ...))
arm = planner.plan_arm()                               # ~0.3 s: {"traj": N × 6 joint angles, "dt": s} or None
```

### Many candidate grasps

[`demo_multi_grasp.py`](demo_multi_grasp.py) runs this whole flow for one capture with N grasps (the executed one, the
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

### From the command line

```bash
python run_demo.py                                                   # test_maps/slam.npz + test_maps/request.json
python run_demo.py --npz map.npz --request request.json              # one request, solvers built for it
python run_demo.py --lab-map /maps/map803_0930 --request request.json
python run_demo.py --lab-map /maps/map803_0930 --serve < requests.txt   # long-running, see §2
python run_demo.py --lab-map /maps/map803_0930 --request arrived.json --arm   # arm trajectory only (after arriving)
```

Every request prints one line `@@JSON@@{...}` (§4).

---

## 2. Deploying on the robot

Run it as **one long-running process**. Building the solvers (~14 s on the Thor) and processing a SLAM map the first
time (~3.8 s) happen once at startup; each request then only writes its obstacles into the existing solvers.

| When | What | Thor |
|---|---|---|
| startup, once | `planner = BasePlacementPlanner(cfg)` | ~14 s |
| startup, once per map | `load_lab_map(map_dir)`: processed once, saved to `.map_cache/` (`$CUROBO_MAP_CACHE` to move it), read afterwards | ~3.8 s first time, then ~7 ms |
| per wrist capture | `grasp_lab.fuse(...)` (later: read the teammate's merged map instead) | ~0.19 s |
| per candidate grasp | `scene_from_slam(...)` → `set_scene(scene)` → `plan()` | 0.18 s + 0.3–0.8 s (27/27 station); ~1.5 s impossible; 5–7 s fallback |
| after arriving, when needed | new capture → `set_scene` → `plan_arm()` | ~0.3 s |

Two ways to connect:

1. **In-process Python calls** (the lab's code imports this package): exactly §1, or `start()` / `merge()` / `solve()` /
   `pick()` in `demo_multi_grasp.py`.
2. **A separate service process**: `python run_demo.py --serve` reads one request per stdin line (a `request.json` path or
   the request itself as JSON; optional keys `"lab_map"`, `"npz"`, `"occupancy"` switch maps, each map is read once,
   `"arm": true` plans the arm instead). It prints `ready (setup …s)` on stderr once the solvers are built and one
   `@@JSON@@{...}` line per request on stdout. A ROS 2 service would replace this stdin loop with a service callback.

On the Thor, inside the lab's `mm-gpu:latest` image (cuRobo is already installed there). Mount warp's kernel cache, or
every start recompiles the kernels:

```bash
cd ~/curobo-base-placement
docker run -d --name base_placement --restart unless-stopped --runtime nvidia --gpus all \
  -v $PWD:/work -w /work \
  -v $PWD/.warp_cache:/root/.cache/warp \
  -v /home/acm/robotic_agent/robotic_system/robot_ws/data/lab:/maps:ro \
  -v /home/acm/mm_system_prod/main_ws/grasp_logs:/grasp_logs:ro \
  --entrypoint bash mm-gpu:latest -c "python3 run_demo.py --lab-map /maps/map803_0930 --serve"
```

Notes:
- Scripts that call these functions need `if __name__ == "__main__":`. Box fitting runs in a pool of CPU processes
  (`map.box_workers`); without the guard it falls back to one core (same result, slower). The first request of a
  process builds its scene more slowly (~0.6 s) because the pool starts then.
- GPU memory: all solvers together stay resident at about 0.5 GB. Capacities are fixed (`profile.max_scene_boxes`,
  `max_near_boxes`, the ESDF box); a request that exceeds them fails loudly instead of silently dropping obstacles.
- Running from a laptop without a GPU: `python bundle.py | ssh <server> 'cd /tmp && <cuRobo python> -'` packs the code,
  config, robot assets and inputs into one script that unpacks to `/dev/shm` on the server and deletes itself afterwards.

---

## 3. Inputs

### 3.1 The SLAM point cloud: `.npz`

| Field | Shape / type | Required | Use |
|---|---|---|---|
| `means3D` | (N, 3) float32, metres | **yes** | point positions in the **map frame** (the frame Kachaka localises in), z up, floor at z ≈ 0 |
| `semantic_ids` | (N,) uint8 | no | class id per point; objects are first separated by class (works without, §5.1) |
| target mask (any name, e.g. `target_mask`) | (N,) bool | recommended | the **target object's points**: they are what is grasped, never an obstacle |
| `logit_opacities` | (N, 1) | no | 3D Gaussian opacity; only points with > 0 are used |
| `<same name>.json` | `{"0": "floor", "1": "table", …}` | no | class names, so a request may name a class |

No obstacle boxes, obstacle lists or grids are needed: obstacles are computed from the points (§5.1). Every point
above the floor, within 1.8 m of the gripper and not part of the target is an obstacle. Other formats (PLY, PCD,
PointCloud2) only need converting:

```python
np.savez("map.npz", means3D=xyz.astype(np.float32))                          # minimum
np.savez("map.npz", means3D=xyz.astype(np.float32), target_mask=is_target)   # recommended
```

### 3.2 The request

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
| `target` (optional) | which points are the target, in order of preference: `{"mask_key": "target_mask"}` (a bool field of the npz, **best**); `{"instance_id": 13}` (lab maps); `{"class": "can"}` (that class within 0.2 m of the gripper; farther ones stay obstacles); omitted → an upright cylinder of radius 7 cm, ±10 cm around the gripper (fine for a small upright can or bottle, unreliable otherwise) |

Why the target matters: it is in the point cloud and the gripper pose is inside it. Not excluded, it becomes an
obstacle and the gripper "collides" at the goal, so IK never succeeds.

### 3.3 The lab's data on the Thor

| What the lab has | How to give it | Notes |
|---|---|---|
| exported 3D semantic map folder (`robot_ws/data/lab/<site>/`) | `--lab-map <dir>` / `lab_map.load_lab_map(dir)` | reads `*_sem_color_fine.npz`, `*_instance_labels.npz`, `*_instance_semantic_table_fine.json`, `alignment_*.yaml`; applies that alignment (the object_query formula) and moves the floor to z = 0 |
| Kachaka's 2D map `occupancy_<site>.yaml/png` | read automatically with `--lab-map`, or `--occupancy` | the base may only park on its free cells (3D maps often lack floor points); Stage1 uses the same map |
| object_query instance ids | `"target": {"instance_id": 13}` | |
| Grasp Lab Global captures (`main_ws/grasp_logs/<run>/`) | `grasp_lab.py` | §8 |

---

## 4. Outputs

`run_demo.py` prints one line `@@JSON@@{...}`; `plan()` returns the same dict. Check `ok` and `reason` first.

| Field | Meaning |
|---|---|
| **`ok`** | true = a station was found |
| **`reason`** | `ok`, or the first step that ran out of candidates (table below) |
| **`chosen_map`** | **(x, y, yaw), map frame → `Navigate_to_pose`** (only when `ok`) |
| `chosen_base` | the same station relative to where Kachaka is now |
| **`at_goal.grasp_pose_in_piper_base`** | the gripper pose in the Piper base frame once Kachaka stands at `chosen_map` → mm_actions `move_arm_to_pose` (`run_demo.py` adds it) |
| `robustness` | fraction of the 27 navigation errors (±5 cm, ±5°) from which the arm still reaches the pose |
| `below_threshold` | true = no station reached `threshold`; this is the fallback (below) |
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

**`reason`** — the first step that removed every candidate:

| reason | Meaning | What upstream can do |
|---|---|---|
| `ok` | a station was found | drive to `chosen_map` |
| `no_ik_solution` | no 9-DoF IK solution in the search window: out of reach, or the gripper pose itself collides | another grasp |
| `outside_cone` | solutions exist, but none keeps the target within the arm's ±30° safety cone (`robot.bearing_limit_rad`) | another grasp, or widen the cone |
| `cannot_park` | solutions exist, but the base can never park there (blocked, or off Kachaka's free cells) | check the map, or another grasp |
| `not_robust` | parkable, but no station reaches `min_accept`; see `max_robustness` | lower `min_accept` at your own risk, or another grasp |
| `arm_plan_failed` | only from `plan_arm()`: no arm trajectory from the stowed arm | check `robot.stow_arm`, retry |

`no_ik_solution` means sampling found nothing (128 seeds × up to 4 rounds), not a proof; a very narrow solution space
can be missed. A colliding gripper pose is also reported this way for now.

**The arm trajectory is a separate step.** `plan()` only answers where to drive. Once Kachaka has arrived and upstream
has looked at the target again, send a new request with the pose actually reached: `set_scene(new scene)` then
`plan_arm()` (base locked, 6-DoF MotionPlanner from the stowed arm). It returns `{"traj": N × 6 (joint1–6, rad), "dt"}`
or `None`. The MotionPlanner is built on the first call (~2–3 s) unless `profile.build_arm_planner: true`.

---

## 5. How it works

```
SLAM npz / lab map folder  (+ wrist capture: grasp_lab.fuse)
   │ scene_io.scene_from_slam(): target points, drivable area, obstacle points
   │   far from the gripper: objects_to_boxes()  (per class → 3D-connected objects → height levels → boxes)
   │   within ±25 cm of the gripper: esdf_grid() (1 cm signed distance field)
   ▼
PlanningScene (base frame = where Kachaka is now)
   │ BasePlacementPlanner.set_scene(): obstacles written straight into the resident solvers' tensors
   │ BasePlacementPlanner.plan():
   │   1 whole_body()     9-DoF IK, one gripper pose → many candidate stations (x, y, yaw)
   │   2 _candidates()    ±30° safety cone; parking_ok(): base clear of obstacles and on drivable cells;
   │                      the current pose is added as the nearest candidate
   │   3 _lm_pass(), _finish()   each station: 27 navigation errors, 6-DoF IK each (cheap LM stage for all,
   │                      the expensive optimiser only where it can change the answer); first 27/27 wins
   ▼
chosen_map → Navigate_to_pose          (later, separately: set_scene(new scene) → plan_arm() → arm trajectory)
```

### 5.1 From points to obstacles

- **Target**: from the mask / instance / class / cylinder (§3.2), grown by 3 cm (`map.target_grow_m`) within the
  target's height range, because segmentation is never exact at the object's edge.
- **Boxes** (everything beyond the ESDF region): points are grouped per class into 3D-connected objects (4 cm voxels);
  each object's footprint is split into **height levels** (a table top at 0.41 m, a board on it at 0.71 m, a cabinet
  at 0.91 m each get their own stacked boxes instead of one block at the tallest height), then covered with few
  rectangles (`decompose`). Boxes are inflated 1 cm horizontally and 0.5 cm vertically. Nothing is filled inside the
  target's footprint above its bottom (`keep_out`). Box fitting runs in parallel per object (identical result).
- **ESDF near the gripper** (`map.esdf_half_m`, default 0.25 m): boxes on a 4 cm grid fill the gap between, say, a
  carton's floor and its back wall; within 50 × 50 × 70 cm of the target they claim 8–33% of the free space
  (measured, §9.4), and the fingers pass within 1–3 cm of the clutter. So within ±0.25 m × ±0.25 m × ±0.35 m of the
  gripper (above 0.15 m; lower obstacles stay boxes for the parking check) the points become a 1 cm signed distance
  field (cuRobo v2 `VoxelData`, float16, negative inside). Every 1 cm column is filled from its lowest to its highest
  point, cut at vertical gaps over 5 cm; the target's footprint is kept free as for the boxes; distances are reduced by
  0.5 cm. Boxes and the grid are evaluated together in the same collision kernel.

![boxes vs ESDF](docs/media/esdf_vs_boxes.png)

*The executed grasp, sliced through the gripper: the boxes (left) fill both sides of the bottle; the ESDF (right) keeps
only the carton floor as an obstacle.*

### 5.2 Search and scoring

- **Whole-body IK**: 128 seeds over base (x, y, yaw) within ±1.5 m of the current pose and the 6 arm joints, all at
  once for the one gripper pose; more rounds (up to 4) if fewer than 15 parkable candidates come back.
- **Filtering**: target within the arm's ±30° safety cone; base spheres clear of the boxes and the footprint on
  drivable cells (Kachaka's 2D map); the current pose is added as a candidate, so a robot that already stands well
  does not move.
- **Robustness**: Kachaka never stops exactly. Each candidate is re-solved as a 6-DoF IK at 27 perturbed base poses
  (every combination of −5/0/+5 cm in x and y and −5/0/+5° in yaw); robustness = the fraction that still reaches the
  pose. Each perturbation has its own copy of the obstacles (cuRobo multi-env), written in closed form.
- **Speed**: only obstacles the arm can reach from a station are loaded (a provable 0.887 m bound around the shoulder;
  farther ones contribute exactly zero). IK runs in two stages: Levenberg–Marquardt (cheap) for every station first,
  then the MPPI + L-BFGS optimiser (99% of the cost) only for perturbations LM left unsolved, and only for stations that
  can still change the answer. Any valid solution from either stage counts.

### 5.3 Frames

| Frame | Definition |
|---|---|
| map | the SLAM / Kachaka map; every input file is in it |
| base | Kachaka **now**: centre of the turning axis on the floor, x forward, z up; planning happens here (`scene_io.map_to_base_xy`, `quat_map_to_base`, `base_to_map_pose`) |
| Piper base | the arm's `base_link`, mounted at (0.08, 0, 0.525) m in the base frame |

---

## 6. Installation

On an x86_64 machine with an NVIDIA GPU, with [uv](https://docs.astral.sh/uv/) (tested from scratch on an RTX PRO 6000, CUDA 12.8 driver):

```bash
uv venv --python 3.12 ~/env_curobo && source ~/env_curobo/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # 1. PyTorch
git clone https://github.com/NVlabs/curobo.git && git -C curobo checkout 4ea7736  # 2. cuRobo v2 (0.8.0)
uv pip install "./curobo[cu12]"
uv pip install -r requirements.txt                                                # 3. numpy, scipy, PyYAML, Pillow
python -c "import curobo, torch; print(curobo.__version__, torch.__version__, torch.cuda.is_available())"   # 0.8.0 2.11.0+cu128 True
```

- With a CUDA 13 driver, use the cu13 PyTorch and `"./curobo[cu13]"`.
- On the Jetson Thor (aarch64), use the lab's `mm-gpu:latest` docker image, which already has cuRobo (§2).
- The first run compiles warp kernels and is slower.
- The robot assets (`robot129.yml`, URDF, meshes) are in `assets/`.

---

## 7. Configuration (`config.yaml`)

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

Options measured but off by default: `robustness.perturb_points: 9` (centre + 8 corners instead of 27) and
`profile.bound_scoring` (§9.5); `map.crop_to_window` (§9.3).

---

## 8. The teammate's aligned wrist scene (Grasp Lab Global)

The coarse SLAM map is blurry near small objects, and it was scanned on a different day than the grasp. A teammate's
**Grasp Lab Global** (in `mm_system_prod`) aligns the wrist camera's view to the SLAM map and saves `map_from_arm`
(Piper base → map) with every capture. **The alignment is the teammate's work; this project only reads it.**
[`grasp_lab.py`](grasp_lab.py) turns a saved capture into our input without changing any of their code:

```bash
python grasp_lab.py --lab-map <map dir> --run <grasp_logs/run> [--run ...] --out <dir> [--top 32] [--drop-stale]
python run_demo.py --serve < <dir>/<run>_requests.jsonl
```

- **Frames**: their map x, y equal ours (same `alignment_*.yaml`); their z keeps the raw plane-normal height, so
  `floor_height_m` (returned by `load_lab_map`) is subtracted.
- **Gripper pose**: a GraspGen grasp frame is `gripper_base` turned −90° about z (checked with FK of the arm's executed
  final pose), so `pinch_center = grasp · Rz(90°) · (0, 0, 0.12 m)`.
- **Merging — a stop-gap until the teammate delivers merged maps**: SLAM points the wrist camera saw through (more than
  5 cm in front of the measured depth) are dropped, and the depth image is added as 1 cm voxels. The target is the
  segmented pixels within 1.5 cm of Grasp Lab's cleaned object cloud; other masked pixels are depth-edge mixed pixels
  and are left out.
- **`--drop-stale` (this test data only)**: map803 was scanned on 9/30 with several bottles on the table; the captures
  are from 10/08 with one. SLAM objects of the target's class mostly inside the camera image are dropped. The videos
  above use this cleaned map. A map scanned together with the capture does not need it.
- **Checks reported, nothing corrected**: wrist points lie a median 1.2 cm from the SLAM points and the bottle stands on
  the SLAM table, so the alignment is good. The arm base implied by `map_from_arm` is 0.42 / 0.47 m high (mounted at
  0.525 m) and tilted 9° in one capture, so the robot's start pose derived from it is only approximate (the
  API → Piper mount is not calibrated yet).

![top views](docs/media/fused_topdown_esdf.png)

*Thor `plan()` output on the cleaned test map: both executed grasps need only 2–7 cm of driving; GraspGen's other
candidates about 0.5 m.*

---

## 9. Validation and measurements

### 9.1 Scenario tests (`validation/run_scenarios.py`)

One condition of the synthetic test scene (`test_maps/`: a can on a table behind a board, a cabinet on the left, walls
behind and to the right) is changed per scenario, to check every `reason` appears where it should. 9/9 pass, with and
without the ESDF:

| Scenario | Change | reason |
|---|---|---|
| `baseline` | none | `ok` (fallback, 0.67) |
| `loose_threshold` | `threshold` 0.5 | `ok` (0.63) |
| `unreachable` | gripper 2 m above the floor | `no_ik_solution` |
| `gripper_in_cabinet` | gripper inside the cabinet | `no_ik_solution` |
| `cone_closed` | safety cone ±0° | `outside_cone` |
| `no_drivable_area` | nothing drivable within 1.2 m | `cannot_park` |
| `sloppy_navigation` | navigation error ±15 cm, ±15° | `not_robust` |
| `arm_plan` | `plan_arm()` at the chosen station | `ok` |
| `bad_stow` | stowed arm all zeros (self-colliding) | `arm_plan_failed` |

### 9.2 Making it fast on the Thor

With real SLAM maps (350–950 boxes instead of 12), a request first took minutes on the Thor: 56 s to build the solvers
and 4–14 s to score each station. In order:

| Change | Effect (Thor, real maps) |
|---|---|
| Only the boxes the arm can reach from a station (provable bound; `validation/check_near_exact.py`: 400k random arm poses never exceed 0.770 m < 0.887 m) | 942 → 135 boxes per station, ~5× faster scoring, identical scores |
| Score = LM seeds ∪ optimiser results (the optimiser was turning valid LM solutions into failures) | scores no longer depend on batch position; 8 seeds pick the same stations as 32 |
| Long-running solvers (`--serve`, `set_scene`) | 56 s per request → once at startup |
| Optimiser only where it can change the answer (`profile.lazy_optimiser`) | scoring 3.2 → 1.3 s, same answer 59/60 |
| Box fitting in parallel (`map.box_workers`) | scene 1.22 → 0.26 s, identical boxes |
| Arm trajectory moved out of `plan()` | request median 1.95 → 1.66 s |
| ESDF near the target (§9.4) | 15 → 20 stations found on 60 requests, 1.60 → 1.48 s |
| Processed-map cache, vectorised wrist merge | map 3.8 s → 7 ms after the first time; merge 0.59 → 0.11 s (identical points) |

Profiling (`validation/profile_plan.py`) showed 99% of scoring time in the optimiser, which rescued only 2.9% of the
perturbations LM missed — hence the lazy optimiser rather than removing it.

### 9.3 Tried and kept off

- **Cropping the map to the arm's reach** (`map.crop_to_window`): the 1.8 m ROI is already about the arm's reach;
  smaller search windows saved ~0.3 s of scene building but found fewer stations and made requests slower overall.
- **A finer box grid near the target** (1–2 cm cells): accurate enough, but 800–6000 boxes instead of ~350 — too slow
  on the Thor. Replaced by the ESDF.

### 9.4 ESDF near the target

Measured with `validation/overcover.py` (volume within 50 × 50 × 70 cm of the target that is farther than 3 cm from any
point, yet inside an obstacle):

![over-coverage](docs/media/esdf_overcover.png)

Thor, same requests with and without the ESDF:

| | Boxes only | Boxes + ESDF |
|---|---|---|
| 3 lab maps (auto_20260826, ec129_0813b, map803_0930), 60 requests: stations found | 15 | **20** |
| … median request with a station | 1.60 s | **1.48 s** |
| fused test map, 57 grasps: stations found | 0 | **28** |
| … median request with a 27/27 station | — | **0.97 s** |

- The 3 requests only the boxes solved: their arm path at the box version's station presses 5–8 mm into points 1.6–2.7 cm
  from the bottle, so the finer model rejects grasps that were really scraping.
- Every one of the 28 fused-map stations was checked independently: `plan_arm()` from the station, then URDF FK spheres
  measured against the **raw points** (not the grid). 28/28 planned, gripper error 0 mm; ignoring points within 1 cm of
  the bottle, 27/28 stay clear the whole way, one touches a point 1.7 cm from the bottle by 1.6 mm (inside the 3 cm
  target allowance).
- Most GraspGen grasps on this capture remain impossible: 8 of 14 representative ones have only 0.2–2.8 cm of real
  clearance, others none.

### 9.5 Robustness check: 9 points vs 27, and bounds (`validation/bench_bound.py`)

| Thor, median | 3 lab maps: 27/27 · fallback (slowest) · total scoring | fused map: 27/27 · fallback (slowest) · total scoring |
|---|---|---|
| 27 points (default) | 1.13 s · 2.92 s (8.6) · 57 s | 0.73 s · 7.09 s (10.6) · 110 s |
| 27 + bounds | 0.78 s · 2.83 s (4.6) · 45 s | 0.53 s · 4.79 s (7.9) · 82 s |
| 9 points | 0.47 s · 1.71 s (3.0) · 25 s | 0.49 s · 2.75 s (4.1) · 44 s |
| 9 + bounds | 0.47 s · 1.53 s (2.0) · 21 s | 0.49 s · 2.38 s (3.2) · 38 s |

- **Bounds** (`profile.bound_scoring`): after LM a station's score lies between what LM solved and that plus the
  unsolved perturbations; stations whose upper bound cannot win skip the optimiser. Same selection rule.
- **9 points** (`robustness.perturb_points: 9`): 45 of 46 stations that passed 9 points also pass all 27; the fallback
  ranking is coarser (3–5 weak picks on the fused map).
- Run-to-run variation is the yardstick: the same settings repeat exactly, but merely reversing the request order makes
  the default pick a different station in 26 of 27 cases (cuRobo's internal random state follows the call sequence).
- Next: "9 points to search, 27 to confirm" should keep today's quality at nearly the 9-point speed.

---

## 10. Files

```
curobo-base-placement/
├─ config.yaml           every assumption and parameter
├─ placement.py          BasePlacementPlanner: solvers, set_scene(), plan(), plan_arm()
├─ scene_io.py           point cloud + request → PlanningScene (target, drivable area, boxes, ESDF grid)
├─ robot_model.py        the 9-DoF cuRobo model (Kachaka x, y, yaw + Piper), safety cone, reach bound
├─ lab_map.py            the lab's exported map folder → map-frame points (cached in .map_cache/)
├─ grasp_lab.py          Grasp Lab Global captures → merged points + requests (§8)
├─ run_demo.py           command line: one request, --serve, --arm
├─ demo_multi_grasp.py   the whole flow with N grasps (§1)
├─ bundle.py             pack everything into one script for a GPU server
├─ make_test_maps.py, test_scene.yaml, test_maps/   the synthetic test scene
├─ validation/           tests and benchmarks; the main code never imports them (validation/README.md)
├─ assets/               robot129.yml (cuRobo config, collision spheres), URDF and meshes
└─ docs/media/           the videos and figures in this README
```

---

## 11. Known limitations and open items

- **Navigation error ±5 cm / ±5° is assumed**; measure Kachaka's real error, which changes how often a perfect
  station exists.
- **Fallback requests are slow** (5–11 s on the Thor) because every candidate is scored; "9 points to search, 27 to
  confirm" (§9.5) is the next step.
- **Grasp orientation is fixed by upstream.** An obstacle right in front of the bottle blocks every grasp from that
  side (the board video works only because a side grasp exists); upstream then has to supply another grasp.
- **Maps**: the SLAM map and the wrist capture of the test data are from different days (`--drop-stale`); merging should
  move to the teammate's complete version. The API → Piper mount must be calibrated (implied arm base 0.42 vs 0.525 m).
- `no_ik_solution` is not a proof; a colliding gripper pose is reported the same way.
- Only the Kachaka body is modelled for the base; the column holding the arm above it is not, and the camera pole on
  the right side is missing from `robot.kachaka.extra_spheres`.
- Space under tables etc. is solid in the box model (the ESDF keeps gaps over 5 cm only near the target).
- cuRobo v2's `collision_checking.validate()` has a bug in this version; parking is checked with numpy instead.
