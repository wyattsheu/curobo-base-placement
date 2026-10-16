# curobo-base-placement

**Where should the Kachaka base stand so its arm can grasp?** Given a 3D SLAM point cloud and a gripper pose decided
upstream, this returns the base pose (x, y, yaw) to drive to, checked so that the arm still reaches that gripper pose
without collisions even if Kachaka stops ±5 cm / ±5° off. Impossible grasps are rejected with a reason.

Kachaka mobile base + AgileX Piper arm, planned with [cuRobo v2](https://github.com/NVlabs/curobo) (9-DoF whole-body IK)
on a Jetson Thor or any CUDA GPU. About 1 s per grasp on the Thor.

| Out of reach: drive there | Beside it: small correction | Board in front: grasp from the side | Box in front: reach over it |
|---|---|---|---|
| ![far](docs/media/far.gif) | ![near](docs/media/near.gif) | ![board](docs/media/board.gif) | ![box](docs/media/box.gif) |
| [mp4](docs/media/far.mp4) | [mp4](docs/media/near.mp4) | [mp4](docs/media/board.mp4) | [mp4](docs/media/box.mp4) |

*The lab's map803 SLAM map with a real wrist capture; the board and the box are added obstacles.*

## Install

Needs PyTorch with CUDA and cuRobo v2 (commit `4ea7736`), then:

```bash
pip install -r requirements.txt
python run_demo.py            # plans the bundled test scene and prints the result
```

On the Jetson Thor, use the lab's `mm-gpu:latest` docker image, which already has cuRobo. Full steps:
[docs/deployment.md](docs/deployment.md#installation-x86_64--nvidia-gpu).

## Use

```python
from robot_model import load_config
from placement import BasePlacementPlanner
from lab_map import load_lab_map
import scene_io

cfg = load_config("config.yaml")
planner = BasePlacementPlanner(cfg)                    # once at startup (~14 s on the Thor)
slam, names, _ = load_lab_map("/maps/map803_0930")     # once per map (~4 s the first time, then cached)

scene = scene_io.scene_from_slam(slam, base_pose, grasp_pos, grasp_quat_wxyz, cfg,   # per grasp (~0.2 s)
                                 target={"instance_id": 13}, occupancy="/maps/map803_0930/occupancy_map803_0930.yaml")
planner.set_scene(scene)
result = planner.plan()                                # ~1 s

if result["ok"]:
    x, y, yaw = result["chosen_map"]                   # → Kachaka Navigate_to_pose
else:
    print(result["reason"])                            # why this grasp cannot be done
```

`base_pose` is Kachaka's current (x, y, yaw) and `grasp_pos` / `grasp_quat_wxyz` the gripper pose (`pinch_center`), all
in the map frame. Build the planner and load maps **once**; every request after that only costs its own ~1 s.

After Kachaka arrives and the target has been seen again: `set_scene(new scene)` then `planner.plan_arm()` returns the
arm trajectory.

**Several candidate grasps?** See [`demo_multi_grasp.py`](demo_multi_grasp.py): it solves each one, rejects the impossible
ones and picks one.

**As a service:** `python run_demo.py --lab-map <map> --serve` reads one request per line and prints one JSON result per
line.

## Result

| Field | Meaning |
|---|---|
| `ok` | a base pose was found |
| `chosen_map` | (x, y, yaw) in the map frame → the navigation goal |
| `robustness` | share of the 27 navigation errors (±5 cm, ±5°) from which the arm still reaches the grasp |
| `reason` | `ok`, or why not: |

| `reason` | Meaning |
|---|---|
| `no_ik_solution` | the gripper pose is out of reach or collides from everywhere |
| `outside_cone` | reachable only with the target outside the arm's ±30° safety cone |
| `cannot_park` | the base cannot park anywhere that reaches it |
| `not_robust` | reachable, but not reliably under navigation error |

## Documentation

| | |
|---|---|
| [docs/usage.md](docs/usage.md) | the full API, many grasps, input and output formats, `config.yaml` |
| [docs/deployment.md](docs/deployment.md) | running it on the Thor as a service, what each step costs, installation |
| [docs/how_it_works.md](docs/how_it_works.md) | point cloud → obstacles (boxes + ESDF), whole-body IK, robustness scoring, frames, files |
| [docs/grasp_lab.md](docs/grasp_lab.md) | reading the teammate's aligned wrist captures (Grasp Lab Global) |
| [docs/benchmarks.md](docs/benchmarks.md) | tests, Thor measurements, what was tried, known limitations |
| [validation/README.md](validation/README.md) | the test and benchmark scripts |
