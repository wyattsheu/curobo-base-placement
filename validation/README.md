# validation/: tests and benchmarks

Test code only. The main code (`run_demo.py`, `placement.py`, `scene_io.py`, …) never imports this folder, and every
change a scenario makes happens in memory (config values, a moved gripper pose, a removed drivable area); `config.yaml`
and `test_maps/` are never modified. Ignore it if you only use the planner.

All need cuRobo and a GPU (see docs/deployment.md). Run from the project root:

| Script | What it does | Results in |
|---|---|---|
| `run_scenarios.py` | one condition of the test scene changed per scenario; checks that every `reason` appears where it should (prints PASS / FAIL) | docs/benchmarks.md |
| `bench_multi_grasp.py` | many gripper poses: scoring seeds / batch size (`sweep`), warm start (`warm`), 20 poses end to end (`e2e`) | docs/benchmarks.md (history) |
| `bench_lab_maps.py` | on the lab's real SLAM maps: do fewer scoring seeds pick stations as good as 32 seeds | docs/benchmarks.md |
| `reach_cull.py` | CPU only: how many obstacle boxes the arm and base can never reach | docs/benchmarks.md |
| `bench_near.py` | same stations, all boxes vs only those within the arm's reach: score differences and time | docs/benchmarks.md |
| `check_near_exact.py` | cuRobo FK check that the dropped boxes can never come within the collision activation distance | docs/benchmarks.md |
| `bench_seeds.py` | scoring seeds / warm start against a 32-seed reference: regret and time | docs/benchmarks.md |
| `bench_roi.py` | smaller search window + map cropped to the arm's reach: scene time, solutions, chosen stations | docs/benchmarks.md |
| `profile_plan.py` | time per step of each request, and within scoring: LM, checks, optimiser, and how much the optimiser rescues | docs/benchmarks.md |
| `bench_lazy.py` | optimiser only where it can change the answer vs before: same stations, scoring time | docs/benchmarks.md |
| `overcover.py` | near the target: volume the boxes / the ESDF claim that is really empty | docs/benchmarks.md |
| `bench_bound.py` | bounded scoring and 9 vs 27 perturbations: time and chosen stations, plus run-to-run variation | docs/benchmarks.md |

```bash
python validation/run_scenarios.py                        # every scenario
python validation/run_scenarios.py no_drivable_area       # only these
python validation/profile_plan.py <lab maps root> auto_20260826 ec129_0813b map803_0930
python validation/bench_bound.py --lab <lab maps root> auto_20260826 ec129_0813b map803_0930
python validation/bench_bound.py --requests <grasp_lab.py output>/all.jsonl
python validation/overcover.py - config.yaml <lab maps root> map803_0930
```

On the Jetson Thor, use the lab's `mm-gpu` image (cuRobo is already installed):

```bash
docker run --rm --runtime nvidia --gpus all -v $PWD:/work -w /work -v $PWD/.warp_cache:/root/.cache/warp \
  -v <lab maps root>:/maps:ro --entrypoint python3 mm-gpu:latest validation/profile_plan.py /maps map803_0930
```
