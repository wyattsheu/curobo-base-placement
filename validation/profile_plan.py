"""Where does a request's time go? Timers around each stage of plan(), and inside scoring around the Levenberg-Marquardt
(LM) seed solve, the check of those seeds, and the MPPI + L-BFGS optimiser. Nothing in placement.py changes: the
instrumented solve_envs below repeats its logic with timers. Needs cuRobo + GPU.

  python validation/profile_plan.py <maps_root> [site ...]

Same requests as bench_roi.py (first graspable object, 10 grasps, Kachaka starting 0.6–0.9 m and 0.9–1.3 m away),
today's config, one long-running planner.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from robot_model import load_config, in_safety_cone  # noqa: E402
import scene_io  # noqa: E402
from bench_lab_maps import load_site  # noqa: E402
from bench_roi import requests, STARTS  # noqa: E402


def sync():
    torch.cuda.synchronize()


class Prof:
    def __init__(self):
        self.t = defaultdict(float); self.n = defaultdict(int); self.calls = []; self.stations = []

    def timed(self, name, f, *a, **k):
        sync(); t = time.perf_counter(); r = f(*a, **k); sync()
        self.t[name] += time.perf_counter() - t; self.n[name] += 1
        return r


def instrument(pl, prof):
    """solve_envs / robustness with timers; same logic as placement.py."""
    def solve_envs(goal, warm=None, care=None):
        ik, ns = pl.ik_rob, pl.cfg["profile"]["scorer_seeds"]
        g = goal.reorder_links(ik.kinematics.tool_frames)
        seeds = prof.timed("lm_solve", ik.seed_ik_solver.solve_batch, goal_tool_poses=g, return_seeds=ns)
        seeds = seeds.js_solution.position.view(pl.E, ns, -1).clone()
        lm = ik.config.use_lm_seed; ik.config.use_lm_seed = False
        try:
            ok = prof.timed("lm_check", ik.solve_pose, goal, seed_config=seeds, return_seeds=ns, run_optimizer=False)
            ok = ok.success.reshape(pl.E, -1).any(dim=1)
            care_t = torch.ones_like(ok) if care is None else torch.as_tensor(care, device=ok.device)
            need = ~ok & care_t
            lm_ok = ok.clone()
            if bool(need.any()):
                r = prof.timed("optimiser", ik.solve_pose, goal, seed_config=seeds, return_seeds=ns)
                ok |= r.success.reshape(pl.E, -1).any(dim=1)
            prof.calls.append({"envs": int(care_t.sum()), "unsolved_after_lm": int(need.sum()), "optimiser": bool(need.any())})
        finally:
            ik.config.use_lm_seed = lm
        prof._lm_ok = lm_ok.cpu().numpy()
        return ok.cpu().numpy()

    def robustness(bases):
        bases = np.atleast_2d(bases)[:, :3]; n = len(bases); K = len(pl.OFFS)
        fr = (bases[:, None, :] + pl.OFFS[None]).reshape(-1, 3)
        envf = np.vstack([fr, np.repeat(fr[-1:], pl.E - len(fr), 0)]) if len(fr) < pl.E else fr
        pl.load_envs(envf)
        cone = in_safety_cone(pl.cfg, envf, pl.gpos[:2])
        ok = solve_envs(pl.goal_from(envf), care=cone)[:len(fr)]
        lm_ok = prof._lm_ok[:len(fr)]
        c = cone[:len(fr)]
        final = (ok & c).reshape(n, K).mean(1); lm = (lm_ok & c).reshape(n, K).mean(1)
        for b, f_, l in zip(bases, final, lm):
            prof.stations.append({"base": b.tolist(), "final": float(f_), "lm": float(l)})
        return final

    pl.solve_envs = solve_envs
    pl.robustness = robustness
    lm_envs, opt_envs, load = pl._lm_envs, pl._opt_envs, pl.load_envs            # the lazy path (profile.lazy_optimiser)
    pl._lm_envs = lambda *a, **k: prof.timed("lm_solve", lm_envs, *a, **k)
    pl._opt_envs = lambda *a, **k: prof.timed("optimiser", opt_envs, *a, **k)
    pl.load_envs = lambda *a, **k: prof.timed("load_envs", load, *a, **k)


def main(root, sites):
    from placement import BasePlacementPlanner
    cfg = load_config(HERE.parent / "config.yaml")
    pl = BasePlacementPlanner(cfg)
    rows = []
    for site in sites:
        data, objs, occ = load_site(f"{root}/{site}")
        if not objs:
            continue
        o = objs[0]
        for sname, r0, r1 in STARTS:
            for req in requests(cfg, data, o, occ, r0, r1):
                prof = Prof(); instrument(pl, prof)
                g = req["grasp_pose_in_map"]
                t0 = time.perf_counter()
                sc = scene_io.scene_from_slam(data, req["base_pose_in_map"], g["position"], g["quaternion_wxyz"], cfg,
                                              target=req["target"], occupancy=occ)
                t_scene = time.perf_counter() - t0
                pl.set_scene(sc); res = pl.plan(); total = time.perf_counter() - t0
                T = res["timing_s"]
                row = {"site": site, "start": sname, "grasp": req["grasp"], "ok": res["ok"], "reason": res["reason"],
                       "chosen": res.get("chosen_base"), "total_s": total, "scene_s": t_scene,
                       "wholebody_s": T["wholebody_ik_s"] + T["filter_s"], "scoring_s": T["scoring_s"], "arm_plan_s": T["arm_plan_s"],
                       "lm_solve_s": prof.t["lm_solve"], "lm_check_s": prof.t["lm_check"], "optimiser_s": prof.t["optimiser"],
                       "load_envs_s": prof.t["load_envs"], "calls": prof.calls, "stations": prof.stations}
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ("calls", "stations")}, default=float), flush=True)
    print("@@JSON@@" + json.dumps(rows, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
