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
inside it (see [Deploying on the Thor](#deploying-on-the-thor)).

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

```python
from robot_model import load_config
from placement import BasePlacementPlanner
from lab_map import load_lab_map
import scene_io

cfg = load_config("config.yaml")
planner = BasePlacementPlanner(cfg)                    # once at startup (~14 s on the Thor)
slam, names, _ = load_lab_map("/maps/map803_0930")     # once per map (~4 s the first time, then cached: ~7 ms)
occupancy = "/maps/map803_0930/occupancy_map803_0930.yaml"   # Kachaka's 2D map: where the base may park

# per grasp
scene = scene_io.scene_from_slam(slam, base_pose, grasp_pos, grasp_quat_wxyz, cfg,
                                 target={"instance_id": 13}, occupancy=occupancy)    # ~0.2 s
planner.set_scene(scene)
result = planner.plan()                                # ~0.3–1 s

if result["ok"]:
    x, y, yaw = result["chosen_map"]                   # → Kachaka Navigate_to_pose
else:
    print(result["reason"])                            # why this grasp cannot be done (table below)
```

- `base_pose` = Kachaka's current (x, y, yaw); `grasp_pos`, `grasp_quat_wxyz` = the gripper pose (`pinch_center`, the
  point between the fingers; its +z is the approach direction). All in the map frame, metres and radians.
- `target` tells which points are the object being grasped, so it is not treated as an obstacle: an instance id of the
  lab map, `{"class": "bottle"}`, or `{"mask_key": ...}` for a boolean mask in the point cloud.
- **Build the planner and load maps once.** Every request after that only costs its own time.
- **Arm motion is a separate step.** After Kachaka arrives and the target has been seen again:
  `planner.set_scene(new scene)`, then `planner.plan_arm()` returns the joint trajectory (~0.3 s).
- **Several candidate grasps?** [`demo_multi_grasp.py`](demo_multi_grasp.py) solves each one, rejects the impossible
  ones and picks one; it also shows how to merge a wrist capture into the map.
- **Command line / service:** `python run_demo.py --lab-map <map> --request request.json` for one request;
  `--serve` keeps the planner running and answers one JSON request per stdin line.

### What happens per request (Thor)

| Step | What | Time |
|---|---|---|
| startup, once | build every cuRobo solver and warm up | ~14 s |
| once per map | process the SLAM map (cached afterwards) | ~3.8 s, then ~7 ms |
| per wrist capture | merge what the wrist camera saw into the map | ~0.19 s |
| per grasp: scene | point cloud → obstacles near the grasp (boxes + a 1 cm distance field) | ~0.18 s |
| per grasp: plan | whole-body IK → candidate base poses → test each against 27 navigation errors | 0.3–0.8 s with a good spot; ~1.5 s if impossible; 5–7 s if only weaker spots exist |
| after arriving | arm trajectory (`plan_arm()`) | ~0.3 s |

## Result

| Field | Meaning |
|---|---|
| `ok` | a base pose was found |
| `chosen_map` | (x, y, yaw) in the map frame → the navigation goal |
| `robustness` | share of the 27 navigation errors (±5 cm, ±5°) from which the arm still reaches the grasp (1.0 = all) |
| `below_threshold` | true if no spot survives all 27 and this is the best one (at least half) |
| `reason` | `ok`, or why not (below) |

| `reason` | Meaning | What upstream can do |
|---|---|---|
| `no_ik_solution` | the gripper pose is out of reach or collides from everywhere | another grasp |
| `outside_cone` | reachable only with the target outside the arm's ±30° safety cone | another grasp |
| `cannot_park` | the base cannot park anywhere that reaches it | check the map, another grasp |
| `not_robust` | reachable, but not reliably under navigation error | another grasp |

## Deploying on the Thor

Run it as one long-running process inside the lab's image, so the 14 s startup happens once:

```bash
docker run -d --name base_placement --restart unless-stopped --runtime nvidia --gpus all \
  -v $PWD:/work -w /work -v $PWD/.warp_cache:/root/.cache/warp \
  -v /home/acm/robotic_agent/robotic_system/robot_ws/data/lab:/maps:ro \
  --entrypoint bash mm-gpu:latest -c "python3 run_demo.py --lab-map /maps/map803_0930 --serve"
```

Details, notes and the service protocol: [docs/deployment.md](docs/deployment.md). Plugging it into mm_system (image, callers, file formats): [docs/integration.md](docs/integration.md).

## Documentation

| | |
|---|---|
| [docs/integration.md](docs/integration.md) | **for the mm_system team:** which docker image, how other processes call it, exact input / output formats, open points to agree on |
| [docs/usage.md](docs/usage.md) | the full API, many grasps, input and output formats, `config.yaml` |
| [docs/deployment.md](docs/deployment.md) | running it on the Thor as a service, what each step costs |
| [docs/how_it_works.md](docs/how_it_works.md) | point cloud → obstacles (boxes + ESDF), whole-body IK, robustness scoring, frames, files |
| [docs/grasp_lab.md](docs/grasp_lab.md) | reading the teammate's aligned wrist captures (Grasp Lab Global) |
| [docs/benchmarks.md](docs/benchmarks.md) | tests, Thor measurements, what was tried, known limitations |
| [validation/README.md](validation/README.md) | the test and benchmark scripts |
