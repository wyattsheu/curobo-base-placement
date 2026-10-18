# Integrating with mm_system

This page covers what is needed to plug the base-placement planner into the robot system on the Jetson Thor:

- where it can run;
- how other processes can call it;
- the exact input and output formats;
- the points both sides still need to agree on.

It was written after reading the system as it is on the Thor (`/home/acm/mm_system_prod`, the running containers and
images) on 2026-10-18. Nothing in mm_system was changed.

**Status:**

| Part | State |
|---|---|
| Python API | implemented; this is what all the Thor measurements use |
| stdin/stdout JSON service (`run_demo.py --serve`) | implemented |
| ROS 2 entry point (§2) | **proposal only**; to be agreed |
| live wrist-capture input (§4, item 8) | **proposal only**; to be agreed |

## 1. Where it runs

The planner needs cuRobo and a GPU. On the Thor, use **`mm-gpu`**, the same image as `mm_actions_node`. There is
nothing extra to install: the image already has every package this project uses. All Thor numbers in this repository
were measured inside it.

Versions checked inside the image:

| | Version |
|---|---|
| image | `mm-gpu:latest`, built 2026-09-28 (`sha256:f5f46c5c…`) from `Dockerfile.gpu` |
| Python | 3.10.12 |
| cuRobo | 0.8.0.post1.dev43, commit `78fd485` (`CUROBO_COMMIT` in `Dockerfile.gpu`) |
| PyTorch | 2.12.0+cu130 |
| numpy / scipy / PyYAML / Pillow | 1.26.4 / 1.15.3 / 6.0.3 / 12.3.0 |
| ROS 2 | Humble |

Other images on the Thor will not work. `robotic_agent-system` (decision_maker, kachaka_nav) has no cuRobo, and its
torch does not see the GPU. The vLLM, SLAM and gateway images are unrelated.

What the container needs:

| Item | Mount | Notes |
|---|---|---|
| this repository | read-only | code, `config.yaml`, robot model in `assets/` |
| the map folder | read-only | §3.4 |
| `.map_cache/` | writable | processed maps; or set `$CUROBO_MAP_CACHE` |
| warp kernel cache `/root/.cache/warp` | writable | without it, every start recompiles the kernels |

**Check an image before relying on it** (two commands, about a minute):

```bash
docker run --rm --runtime nvidia --gpus all -v $PWD:/work -w /work --entrypoint bash mm-gpu:latest -c '
  python3 -c "import curobo, torch, warp, scipy, yaml, PIL; print(curobo.__version__, torch.__version__, torch.cuda.is_available())"
  python3 run_demo.py | grep -o "\"ok\": [a-z]*"'
# expected:  0.8.0.post1.dev43 2.12.0+cu130 True
#            "ok": true
```

If `Dockerfile.gpu` ever changes `CUROBO_COMMIT` or torch, also run `python3 validation/run_scenarios.py` in the new
image. All 8 scenarios should pass before the planner is used on the robot.

## 2. How other processes call it

### Rules that hold whatever the transport

- **One planner per process, built once.** It takes ~14 s and keeps ~0.5 GB of GPU memory. Do not build it per
  request.
- **One request at a time.** The planner holds one scene; a `set_scene()` + `plan()` pair must not interleave with
  another. Queue requests rather than running several planners: a second process costs another 14 s and 0.5 GB, and
  both get slower on the shared GPU.
- **Call it from the thread that built it.** The arm-trajectory planner uses CUDA graphs. This is the same rule as
  `GraspStack.run()` in mm_actions.
- Box fitting runs in a pool of CPU processes started with `spawn`. This is safe inside a ROS node. A plain script
  needs `if __name__ == "__main__":`.

### What exists today

1. **Python:** `BasePlacementPlanner(cfg)` once, then `scene_io.scene_from_slam(...)`, `set_scene()` and `plan()` per
   grasp ([usage.md](usage.md)).
2. **A service process:** `python3 run_demo.py --lab-map <map> --serve`.
   - Input: one request per stdin line (§3.2).
   - Output: one `@@JSON@@{...}` line per request on stdout.
   - Only the process that started it can talk to it.

### Proposal: a ROS 2 action in its own container (recommended, not implemented)

- **Container:** the `mm-gpu` image, `--network host`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (as `run.sh`).
- **Action:** one ROS 2 action, `/base_placement`.
- **Action type:** the existing **`mm_interface/action/TaskCommand`**, so no new message type is needed.
  - Goal: `command` = the request JSON (§3.2).
  - Result: `success` = `ok`, `message` = the result JSON (§3.6).
  - Feedback: the current step.

This is the same convention the `standpoint` action already uses, and decision_maker already sends JSON TaskCommands.
A separate process keeps the planner's GPU memory, CUDA graphs and 14 s startup out of `mm_actions_node`, and it can be
restarted on its own.

The other option is to call the planner **inside `mm_actions_node`** (for example behind `actions/standpoint.py`).
That avoids IPC, but it:

- adds 14 s and 0.5 GB to the arm node;
- requires every call to go through the GPU worker thread;
- changes mm_actions code.

That choice belongs to the mm_actions team.

## 3. Input and output formats

### 3.1 Frames and units

| Frame | Definition |
|---|---|
| **map** | Kachaka's 2D map: the frame of `get_robot_pose()` / `move_to_pose()`, `/user_pose`, and the occupancy map. z is up and the floor is z = 0. |
| **Kachaka base** | the turning centre on the floor, x forward |
| **Piper base** | the arm's `base_link`; set by `robot.arm_mount_xyz` in `config.yaml`, currently (0.08, 0, 0.525) m in the Kachaka base frame (see §4) |

Units:

- Lengths are in metres and angles in radians.
- Quaternions are ordered **w, x, y, z** and must be unit length.
- A pose (x, y, yaw) gives yaw in radians, counter-clockwise from the map's +x axis.

### 3.2 The request (one JSON object)

```json
{"base_pose_in_map": [2.341, 0.297, 1.179],
 "grasp_pose_in_piper_base": {"position": [0.43, -0.02, -0.06], "quaternion_wxyz": [0.5, -0.5, 0.5, 0.5]},
 "target": {"class": "bottle"}}
```

(The numbers are only an illustration of the format.)

| Field | Required | Type | Meaning |
|---|---|---|---|
| `base_pose_in_map` | yes | [x, y, yaw] | where Kachaka is **now** |
| `grasp_pose_in_map` | one of the two | {position [3], quaternion_wxyz [4]} | the decided gripper pose (§3.3) in the map frame |
| `grasp_pose_in_piper_base` | one of the two | same | the same pose in the Piper base frame of the robot standing at `base_pose_in_map`, as mm_actions holds it |
| `target` | recommended | object | which points are the object being grasped, so they are not obstacles: `{"instance_id": 13}` (lab-map instance), `{"class": "bottle"}` (that class within 0.2 m of the gripper), or `{"mask_key": "<npz field>"}`. If it is missing, an upright cylinder of radius 7 cm around the gripper is used. |

Extra keys for `--serve` (and the proposed action):

| Key | Meaning |
|---|---|
| `"lab_map": "<dir>"` | switch to another map; each map is read once |
| `"npz": "<file>"` | same, for a plain npz map |
| `"occupancy": "<yaml>"` | same, for the 2D map |
| `"arm": true` | plan the arm trajectory instead of a base pose |

### 3.3 Which gripper pose: `pinch_center`

The requested pose is **`pinch_center`**:

- its origin is the point between the fingertips, **0.12 m** along `gripper_base` z (`robot.tool_offset_m`; the 0.125
  in `assets/robot129.yml` is overridden by `config.yaml`);
- its +z is the approach direction.

It is **not** the `grasp_frame` used by mm_actions' cuRobo and GraspGenX:

- `grasp_frame` = `link6` turned −90° about z, with no offset (`curobo_piper.yml`);
- `gripper_base` coincides with `link6` (URDF `joint6_to_gripper_base` is identity).

So a grasp `T` (4×4, `grasp_frame`) becomes:

```python
Rz90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
Tz   = np.eye(4); Tz[2, 3] = 0.12
pinch = T @ Rz90 @ Tz                      # = grasp_lab.pinch_pose(T, 0.12); stays in T's frame
```

This was checked against FK of the arm's executed final pose in the Grasp Lab logs. Send the final grasp pose, i.e. the
one mm_actions would plan to (after `depth_offset_m`), not the pre-grasp.

### 3.4 Maps

Use **one of** the following.

**A. The lab's exported map folder** (`robot_ws/data/lab/<site>/`, e.g. `map803_0930`): `--lab-map <dir>` /
`lab_map.load_lab_map(dir)`. Files are found by pattern, so exactly one of each must exist:

| Pattern | What is read |
|---|---|
| `*_sem_color_fine.npz` | `points` (N, 3), 3D map frame (y vertical) |
| `*_instance_labels.npz` | `centers` (M, 3), `cluster_id` (M,) |
| `*_instance_semantic_table_fine.json` | `labels`, `voxel_size`, `instances[{inst_id, class_id, …}]` |
| `alignment_*.yaml` | `plane_fit` (`mu`, `basis_e1`, `basis_e2`, `normal_n`) and `sim2` (`s`, `R`, `t`): 3D map → map frame |
| `occupancy_*.yaml` + `.png` | the 2D map (format C below) |

The floor is moved to z = 0, and the result is cached in `.map_cache/`.

**B. A plain point cloud `.npz`:** `--npz <file>`.

| Field | Required | Type | Meaning |
|---|---|---|---|
| `means3D` | yes | (N, 3) float32 | points in the **map frame**, floor at z ≈ 0 |
| `semantic_ids` | no | (N,) uint8 | class id per point |
| a target mask, e.g. `target_mask` | no | (N,) bool | the target's points |
| `<same name>.json` (a separate file) | no | `{"0": "floor", …}` | class names |

mm_actions' `config/maps/<site>/map_cloud.npz` converts in one line: `points_mm` is (N, 3) int16 in mm, map frame,
z above the floor.

```python
np.savez("map.npz", means3D=(np.load("map_cloud.npz")["points_mm"] / 1000.0).astype(np.float32))
```

**C. The 2D occupancy map:** a ROS `map_server` yaml + png with `image`, `resolution`, `origin` and values 0 / 205 /
254. The base only parks on free cells.

- It must be the map Kachaka is localised in. For map803 the lab export and mm_actions' `config/maps/map803_0930` hold
  the same map (Map803_3, `map_id 9ce79b5a…`, identical png).
- Compare `source.map_id` with Kachaka's `get_current_map_id()` before driving, as the standpoint action does.

### 3.5 Grasp Lab Global run folder (optional)

`grasp_lab.load_capture(run, floor_height_m, map_dir)` reads these files from `main_ws/grasp_logs/<run>/`:

| File | What is read |
|---|---|
| `global_pose.json` | `map_from_arm` (4×4, Piper base → map; **required**), `aligned_map_folder` (must match the map folder name) |
| `clouds.npz` | `scene_pts`, `object_pts` (Piper base frame), `base_T_cam` |
| `summary.json` | `label`, and `intrinsics` (`fx`, `fy`, `cx`, `cy`) from the stages |
| `input_depth.png` | uint16 depth in mm |
| `seg_mask.png` | target mask (> 0) |
| `grasps.npz` | `poses` (K, 4, 4) in `grasp_frame`, `confidence` (K,) |
| `plan.npz` | `chosen_grasp` (4×4), the executed grasp |

### 3.6 The result

| Field | Meaning |
|---|---|
| `ok` | true if a base pose was found |
| `reason` | `ok`, `no_ik_solution`, `outside_cone`, `cannot_park`, `not_robust`, or (arm step) `arm_plan_failed`; with `--serve`, `error` plus an `error` message |
| `chosen_map` | [x, y, yaw], map frame: the goal for `move_to_pose` |
| `at_goal.grasp_pose_in_piper_base` | the same gripper pose (`pinch_center`) seen from the arm once Kachaka stands at `chosen_map` |
| `robustness` | fraction of the 27 navigation errors (±5 cm, ±5°) from which the arm still reaches the pose |
| `below_threshold` | true if this is a fallback station (robustness below 1.0, at least 0.5) |
| `timing_s` | seconds per step |
| `arm` | only with `"arm": true`: `{"traj": N × 6 joint1–6 (rad), "dt": s}` from the stowed arm |

The meaning of each `reason` and what to do about it is in [usage.md](usage.md#outputs).

## 4. To agree on at handover

| # | Item | Our side now | What we saw in mm_system | Needed |
|---|---|---|---|---|
| 1 | **Arm mount on Kachaka** | (0.08, 0, 0.525) m, from the lab URDF | `motion/standpoint.py`: (0.10, 0, 0.50); `map_from_arm` in the Grasp Lab logs puts the arm base at 0.42–0.47 m and tilted 9° in one capture | one measured value used by both. It also moves `grasp_pose_in_piper_base` ↔ map |
| 2 | Stowed arm (driving pose and start of the arm trajectory) | `robot.stow_arm` = [0, 0.2, −0.6, 0, 0.8, 0] (the old home) | `execution.home_q` = [0, 0.486, −1.113, 0.082, 1.090, 0.014] | which pose the arm is in while driving |
| 3 | Navigation error to be robust against | ±5 cm, ±5° (assumed) | — | measure Kachaka's arrival error |
| 4 | Robot pose source | the caller sends `base_pose_in_map` | the standpoint action reads `robot_pose` from the command or `/user_pose`; decision_maker uses TF `map → base_link` | which one the caller sends; it must be the turning centre in the map frame |
| 5 | Map | the lab export folder | `config/maps/<site>` (2D map + `map_cloud.npz`) | which folder is mounted; `map_id` check before driving |
| 6 | Which process calls it, and how | §2 | decision_maker → TaskCommand JSON → mm_actions | ROS 2 action in a separate container, or inside mm_actions |
| 7 | Image | `mm-gpu:latest` (2026-09-28) | `run.sh` rebuilds `mm-gpu` from `Dockerfile.gpu` | re-run the check in §1 after a rebuild |
| 8 | Live wrist capture | read from a saved run folder (§3.5) | the standpoint action has `object_pts` / `scene_pts` (Piper base frame) in memory | an in-memory request: those points + `map_from_arm` + grasps |
