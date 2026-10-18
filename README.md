# curobo-base-placement

**Where should the Kachaka base stand so its arm can grasp?** Upstream has already decided what to grasp and the gripper's
final pose. Given the 3D SLAM point cloud and that gripper pose, this project returns the base pose (x, y, yaw) for
Kachaka's `Navigate_to_pose`, checked so that from there the arm still reaches **the same gripper pose** without
collisions — even if Kachaka stops ±5 cm / ±5° off. Grasps that cannot be done are rejected with a reason.

- Robot: Kachaka mobile base + AgileX Piper 6-DoF arm (Robot129).
- Planner: [cuRobo v2](https://github.com/NVlabs/curobo), 9-DoF whole-body IK (base x, y, yaw + 6 arm joints).
- Runs on the Jetson Thor or any CUDA GPU: about **1 s per grasp** on the Thor once started.

It does not choose grasps, plan the navigation path, or build / align maps.

## Examples

The lab's map803 SLAM map with a real wrist capture (Grasp Lab Global). The board and the box are added obstacles.
Grey = SLAM points, blue = wrist camera, red = the bottle, green rectangle = the base pose we output.

| Starting 1.5–1.9 m away: drive to the right spot | Starting 1.0–1.4 m away |
|:---:|:---:|
| <img src="docs/media/far2.gif" width="430"> | <img src="docs/media/far.gif" width="430"> |
| a side grasp; drives 1.45 m to a 27/27 spot · [mp4](docs/media/far2.mp4) | the executed grasp; drives 0.34 m · [mp4](docs/media/far.mp4) |
| **Already beside it: a small correction** | **Board in front: reject the front grasp, use a side grasp** |
| <img src="docs/media/near.gif" width="430"> | <img src="docs/media/board.gif" width="430"> |
| 21/27 where it stands, moves 0.10 m to 27/27 · [mp4](docs/media/near.mp4) | front grasp rejected (no IK solution); side grasp after driving 0.51 m · [mp4](docs/media/board.mp4) |
| **Box in front: move, then reach over it** | |
| <img src="docs/media/box.gif" width="430"> | |
| 9/27 where it stands, moves 0.17 m to 27/27 · [mp4](docs/media/box.mp4) | |

## Installation

You need an NVIDIA GPU with PyTorch (CUDA) and **cuRobo v2**, then this project's small requirements.

**On the Jetson Thor:** the lab's `mm-gpu:latest` docker image already has cuRobo; nothing to install. Run everything
inside it (see [Loading everything when the system starts](#loading-everything-when-the-system-starts)).

**On an x86_64 machine with an NVIDIA GPU** (tested from scratch on an RTX PRO 6000 with a CUDA 12.8 driver), using
[uv](https://docs.astral.sh/uv/):

```bash
# 1. a Python 3.12 environment
uv venv --python 3.12 ~/env_curobo
source ~/env_curobo/bin/activate

# 2. PyTorch with CUDA (CUDA 12.8 build; with a CUDA 13 driver use the cu13 build)
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 3. cuRobo v2, pinned to the commit this project was tested with (0.8.0)
git clone https://github.com/NVlabs/curobo.git
git -C curobo checkout 4ea7736
uv pip install "./curobo[cu12]"          # "./curobo[cu13]" with a CUDA 13 driver

# 4. this project
git clone https://github.com/wyattsheu/curobo-base-placement.git curobo-base-placement && cd curobo-base-placement
uv pip install -r requirements.txt       # numpy, scipy, PyYAML, Pillow

# 5. check: should print 0.8.0 2.11.0+cu128 True
python -c "import curobo, torch; print(curobo.__version__, torch.__version__, torch.cuda.is_available())"

# 6. try it: plans the bundled test scene (a can on a table behind a board) and prints the result
python run_demo.py
```

The first run compiles cuRobo's warp kernels and is slower; later runs reuse them. The robot model (URDF,
meshes, collision spheres) is already in `assets/`.

## Usage

One class, one call. It runs inside the lab's `mm-gpu` image (the image of `mm_actions_node`): nothing to install,
just put this folder on the Python path.

```python
from base_placement import BasePlacement

bp = BasePlacement("/maps/map803_0930")       # once at startup (~14 s on the Thor): cuRobo + the map

# whenever a grasp is planned — the inputs are what mm_actions already has
r = bp.find_standpoint(
    robot_pose=(x, y, yaw),                   # Kachaka now, map frame (KachakaApiClient.get_robot_pose())
    grasps=grasp_poses,                       # GraspGenX grasps: 4x4, Piper base frame, best first
    object_pts=clouds.object_pts,             # optional: the target from the wrist camera, Piper base frame
    scene_pts=clouds.scene_pts)               # optional: the rest of the wrist view, Piper base frame

if r["ok"]:
    kachaka.move_to_pose(*r["pose"])          # (x, y, yaw), map frame
    # after arriving: capture again and grasp as usual
else:
    print(r["reason"])
```

It tries the grasps in order and stops at the first one with a base pose that survives every navigation error;
about **1 s per grasp** on the Thor. Use one `BasePlacement` per process and call it from one thread.

| Result | Meaning |
|---|---|
| `ok` | a base pose was found |
| `pose` | (x, y, yaw) in the map frame → `move_to_pose` |
| `grasp_index` | which of `grasps` it is for |
| `robustness` | share of the 27 navigation errors (±5 cm, ±5°) from which the arm still reaches the grasp (1.0 = all) |
| `fallback` | true if no pose survives all 27 and this is the best one (at least half) |
| `reason` | `ok`, or why not: `no_ik_solution` (out of reach), `outside_cone` (target outside the arm's ±30° cone), `cannot_park` (no free floor there), `not_robust` (reachable, but not reliably) |
| `time_s`, `tried` | time, and the outcome of each grasp tried |

**One thing to check on the robot:** the arm mount in `config.yaml` (`robot.arm_mount_xyz`, now (0.08, 0, 0.525) m)
turns the Piper-frame inputs into the map. `motion/standpoint.py` in mm_actions uses (0.10, 0, 0.50); the measured value
should be used in both. If Grasp Lab Global has aligned the capture, pass its `map_from_arm=` instead and the mount is
not used.

### Loading everything when the system starts

All the slow parts happen inside `BasePlacement(...)`: building the cuRobo solvers, loading the map and starting the
CPU workers. Create it **once when the arm node starts**, next to the grasp models. After that, every call is about 1 s.

The arm node already runs all GPU work on one worker thread (`GraspStack.run`). Use the same thread for this:

```python
# when mm_actions_node starts (e.g. right after GraspStack is loaded)
sys.path.insert(0, "/workspace/curobo-base-placement")
from base_placement import BasePlacement
bp = stack.run(BasePlacement, "/maps/map803_0930")               # ~15 s, once

# in an action
r = stack.run(bp.find_standpoint, robot_pose, grasps, clouds.object_pts, clouds.scene_pts)
```

Add these to the container's `docker run` (e.g. `run.sh`). The two caches must survive restarts:

```bash
  -v ~/curobo-base-placement:/workspace/curobo-base-placement \
  -v /home/acm/robotic_agent/robotic_system/robot_ws/data/lab:/maps:ro \
  -v ~/curobo-base-placement/.warp_cache:/root/.cache/warp \
  -e CUROBO_MAP_CACHE=/workspace/curobo-base-placement/.map_cache \
```

| Startup on the Thor (`bp.startup_s`) | Normal | Very first time |
|---|---|---|
| cuRobo solvers | ~11 s | ~56 s (GPU kernels are compiled once into the warp cache) |
| map | ~7 ms | ~3.9 s (processed once into `.map_cache/`) |
| CPU workers | ~0.2 s | ~0.2 s |
| imports and the rest | ~3 s | ~3 s |
| **total** | **~15 s** | ~65 s |

- **Switching maps:** `bp.set_map("/maps/<other map>")`. It takes milliseconds once that map has been seen.
- **GPU memory:** about 0.5 GB stays in use. One instance per process is enough; requests are answered one at a time.

Lower-level calls (one grasp pose in the map frame, the arm trajectory, other map formats) are in
[docs/usage.md](docs/usage.md).

## Documentation

| | |
|---|---|
| [docs/usage.md](docs/usage.md) | the full API, many grasps, input and output formats, `config.yaml` |
| [docs/deployment.md](docs/deployment.md) | running it on the Thor as a service, what each step costs |
| [docs/how_it_works.md](docs/how_it_works.md) | point cloud → obstacles (boxes + ESDF), whole-body IK, robustness scoring, frames, files |
| [docs/grasp_lab.md](docs/grasp_lab.md) | reading the teammate's aligned wrist captures (Grasp Lab Global) |
| [docs/benchmarks.md](docs/benchmarks.md) | tests, Thor measurements, what was tried, known limitations |
| [validation/README.md](validation/README.md) | the test and benchmark scripts |
