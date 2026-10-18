# Deployment

Run the planner as **one long-running process**. Building the solvers (~14 s on the Thor) and processing a SLAM map the
first time (~3.8 s) happen once at startup; each request then only writes its obstacles into the existing solvers.

| When | What | Thor |
|---|---|---|
| startup, once | `planner = BasePlacementPlanner(cfg)` | ~14 s |
| startup, once per map | `load_lab_map(map_dir)`: processed once, saved to `.map_cache/` (`$CUROBO_MAP_CACHE` to move it), read afterwards | ~3.8 s first time, then ~7 ms |
| per wrist capture | `grasp_lab.fuse(...)` (later: read the teammate's merged map instead) | ~0.19 s |
| per candidate grasp | `scene_from_slam(...)` → `set_scene(scene)` → `plan()` | 0.18 s + 0.3–0.8 s (27/27 station); ~1.5 s impossible; 5–7 s fallback |
| after arriving, when needed | new capture → `set_scene` → `plan_arm()` | ~0.3 s |

## Two ways to connect

1. **In-process Python calls**: the lab's code imports this package and follows [usage.md](usage.md), or
   `start()` / `merge()` / `solve()` / `pick()` in [`demo_multi_grasp.py`](../demo_multi_grasp.py).
2. **A separate service process**: `python run_demo.py --serve` reads one request per stdin line (a `request.json` path
   or the request itself as JSON; optional keys `"lab_map"`, `"npz"`, `"occupancy"` switch maps, each map is read once,
   `"arm": true` plans the arm instead). It prints `ready (setup …s)` on stderr once the solvers are built and one
   `@@JSON@@{...}` line per request on stdout. A ROS 2 service would replace this stdin loop with a service callback.

## On the Jetson Thor

Use the lab's `mm-gpu:latest` image, the image of `mm_actions_node` (checked 2026-10-18: Python 3.10, cuRobo
0.8.0.post1.dev43 / commit `78fd485`, torch 2.12.0+cu130 — nothing to install). Mount warp's kernel cache, or every
start recompiles the kernels:

```bash
git clone https://github.com/wyattsheu/curobo-base-placement.git ~/curobo-base-placement
cd ~/curobo-base-placement
docker run -d --name base_placement --restart unless-stopped --runtime nvidia --gpus all \
  -v $PWD:/work -w /work \
  -v $PWD/.warp_cache:/root/.cache/warp \
  -v /home/acm/robotic_agent/robotic_system/robot_ws/data/lab:/maps:ro \
  -v /home/acm/mm_system_prod/main_ws/grasp_logs:/grasp_logs:ro \
  --entrypoint bash mm-gpu:latest -c "python3 run_demo.py --lab-map /maps/map803_0930 --serve"
```

## Notes

- Scripts that call these functions need `if __name__ == "__main__":`. Box fitting runs in a pool of CPU processes
  (`map.box_workers`); without the guard it falls back to one core (same result, slower). The first request of a
  process builds its scene more slowly (~0.6 s) because the pool starts then.
- GPU memory: all solvers together stay resident at about 0.5 GB. Capacities are fixed (`profile.max_scene_boxes`,
  `max_near_boxes`, the ESDF box); a request that exceeds them fails loudly instead of silently dropping obstacles.
- The Thor runs in 120 W mode and shares its GPU with other containers; MAXN + `jetson_clocks` would be faster but
  affects everyone on the machine (needs the lab's agreement).
- Without a GPU on your machine: `python bundle.py | ssh <server> 'cd /tmp && <cuRobo python> -'` packs the code,
  config, robot assets and inputs into one script that unpacks to `/dev/shm` on the server and deletes itself afterwards.

## Installation (x86_64 + NVIDIA GPU)

With [uv](https://docs.astral.sh/uv/) (tested from scratch on an RTX PRO 6000, CUDA 12.8 driver):

```bash
uv venv --python 3.12 ~/env_curobo && source ~/env_curobo/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # 1. PyTorch
git clone https://github.com/NVlabs/curobo.git && git -C curobo checkout 4ea7736  # 2. cuRobo v2 (0.8.0)
uv pip install "./curobo[cu12]"
uv pip install -r requirements.txt                                                # 3. numpy, scipy, PyYAML, Pillow
python -c "import curobo, torch; print(curobo.__version__, torch.__version__, torch.cuda.is_available())"   # 0.8.0 2.11.0+cu128 True
```

- With a CUDA 13 driver, use the cu13 PyTorch and `"./curobo[cu13]"`.
- The first run compiles warp kernels and is slower.
- The robot assets (`robot129.yml`, URDF, meshes) are in `assets/`.
