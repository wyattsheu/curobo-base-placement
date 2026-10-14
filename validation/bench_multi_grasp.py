"""Timing for many candidate gripper poses (upstream sends N grasps, we pick one). Needs cuRobo + GPU.

  python validation/bench_multi_grasp.py sweep   # scoring cost and accuracy vs scorer_seeds / scorer_group
  python validation/bench_multi_grasp.py e2e     # N grasps end to end: plan() per grasp vs score-all-then-plan-once

The grasps are around the test scene's can (same position, different approach directions), so the scene, the
search window and the solvers are shared; only the gripper pose changes between grasps.
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scenarios import load_test_scene, cfg_set  # noqa: E402
from robot_model import in_safety_cone  # noqa: E402
from curobo.types import GoalToolPose, Pose  # noqa: E402

N_GRASPS = 20


# ---------------------------------------------------------------- grasps
def quat_from_R(R):
    w = math.sqrt(max(1e-12, 1 + np.trace(R))) / 2
    if w > 1e-3:
        return [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0] * 4; q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def make_grasps(scene, n=N_GRASPS, pitches_deg=(0, 30)):
    """n gripper poses at the can: approach yaw every 360/(n/2) deg, horizontal and tilted 30 deg down.
    Tool z = approach direction; tool x points down-ish (fingers close horizontally), as in the test request."""
    pos = list(scene.grasp_pos_base)
    out = []
    for pitch in [math.radians(p) for p in pitches_deg]:
        for a in np.linspace(0, 2 * math.pi, n // len(pitches_deg), endpoint=False):
            z = np.array([math.cos(a) * math.cos(pitch), math.sin(a) * math.cos(pitch), -math.sin(pitch)])
            x = np.array([0.0, 0.0, -1.0]); x = x - z * (x @ z); x /= np.linalg.norm(x)
            R = np.stack([x, np.cross(z, x), z], 1)
            out.append({"yaw_deg": round(math.degrees(a)), "pitch_deg": round(math.degrees(pitch)),
                        "pos": pos, "quat": quat_from_R(R)})
    return out


def set_grasp(pl, g):
    pl.gpos = np.array(g["pos"], float); pl.gquat = np.array(g["quat"], float)


# ---------------------------------------------------------------- scoring several grasps in one call
def goal_multi(pl, F, P, Qt):
    """Like planner.goal_from, but row i has its own gripper pose (P[i], Qt[i])."""
    from placement import _qmul, _qz
    c, s = np.cos(-F[:, 2]), np.sin(-F[:, 2]); px, py = P[:, 0] - F[:, 0], P[:, 1] - F[:, 1]
    pos = np.stack([c * px - s * py, s * px + c * py, P[:, 2]], 1)
    quat = _qmul(_qz(-F[:, 2]), Qt)
    return GoalToolPose.from_poses({pl.cfg["robot"]["tool_frame"]: Pose(
        position=torch.tensor(pos, device="cuda", dtype=torch.float32),
        quaternion=torch.tensor(quat, device="cuda", dtype=torch.float32))}, num_goalset=1)


def robustness_multi(pl, stations, P, Qt, warm=False):
    """planner.robustness() for up to pl.group stations that may belong to different grasps (P, Qt per station).
    warm: seed each problem with the station's own arm solution from the whole-body IK (other seeds stay random)."""
    stations = np.atleast_2d(stations); bases = stations[:, :3]; n, K = len(bases), len(pl.OFFS)
    fr = (bases[:, None, :] + pl.OFFS[None]).reshape(-1, 3)
    Pe, Qe = np.repeat(P, K, 0), np.repeat(Qt, K, 0)
    pad = pl.E - len(fr)
    envf = np.vstack([fr, np.repeat(fr[-1:], pad, 0)]); Pe = np.vstack([Pe, np.repeat(Pe[-1:], pad, 0)])
    Qe = np.vstack([Qe, np.repeat(Qe[-1:], pad, 0)])
    cy, sy = np.cos(-pl.Y)[None], np.sin(-pl.Y)[None]
    dx = envf[:, None, 0] - pl.C[None, :, 0]; dy = envf[:, None, 1] - pl.C[None, :, 1]
    inv = np.zeros((pl.E, len(pl.names), 7), np.float32)
    inv[:, :, 0] = cy * dx - sy * dy; inv[:, :, 1] = sy * dx + cy * dy; inv[:, :, 2] = -pl.C[None, :, 2]
    ang = envf[:, None, 2] - pl.Y[None]
    inv[:, :, 3] = np.cos(ang / 2); inv[:, :, 6] = np.sin(ang / 2)
    pl.cub.inv_pose[:, :len(pl.names), :7] = torch.from_numpy(inv).to(pl.cub.inv_pose.device)
    seed = None
    if warm:
        from robot_model import BASE_JOINTS, ARM_JOINTS
        jn = list(pl.ik_rob.joint_names)
        A = stations[:, 3:9][:, [ARM_JOINTS.index(j) for j in jn]]
        A = np.repeat(A, K, 0); A = np.vstack([A, np.repeat(A[-1:], pad, 0)])
        seed = torch.tensor(A[:, None, :], device="cuda", dtype=torch.float32)
    r = pl.ik_rob.solve_pose(goal_multi(pl, envf, Pe, Qe), seed_config=seed)
    ok = r.success.reshape(pl.E, -1).any(dim=1).cpu().numpy()[:len(fr)]
    cone = np.array([in_safety_cone(pl.cfg, f[None], p[:2])[0] for f, p in zip(fr, Pe[:len(fr)])])
    return (ok & cone).reshape(n, K).mean(axis=1)


def score_pool(pl, pool, warm=False):
    """pool: list of (grasp, station). Scores in chunks of pl.group, mixing grasps. Returns robustness per item."""
    out = []
    for i in range(0, len(pool), pl.group):
        chunk = pool[i:i + pl.group]
        P = np.array([g["pos"] for g, _ in chunk], float); Qt = np.array([g["quat"] for g, _ in chunk], float)
        out += robustness_multi(pl, np.array([s for _, s in chunk]), P, Qt, warm).tolist()
    return out


# ---------------------------------------------------------------- helpers
def build(cfg0, scene0, **over):
    from placement import BasePlacementPlanner
    cfg, scene = copy.deepcopy(cfg0), copy.deepcopy(scene0)
    cfg_set(**over)(cfg, scene)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter(); pl = BasePlacementPlanner(cfg, scene)
    return pl, time.perf_counter() - t


def sync():
    torch.cuda.synchronize()


def candidates(pl, g, k_max=None, stop_on_zero=False):
    """Whole-body IK (up to wb_rounds_max rounds) + filter, as in plan(); parkable stations nearest first.
    stop_on_zero: give up after a round with no IK solution at all (more rounds only help when there are too few)."""
    set_grasp(pl, g)
    P = pl.cfg["profile"]; Q = np.zeros((0, 9))
    for _ in range(P["wb_rounds_max"]):
        q = pl.whole_body()
        if stop_on_zero and not len(q) and not len(Q):
            return np.zeros((0, 9))
        Q = np.vstack([Q, q])
        _, U, park = pl._candidates(Q)
        if park.sum() >= P["min_candidates"]:
            break
    if not len(U):
        return np.zeros((0, 9))
    S = U[park]
    S = S[np.argsort(np.hypot(S[:, 0], S[:, 1]))]
    return S[:k_max] if k_max else S


# ---------------------------------------------------------------- sweep: scoring cost / accuracy
def sweep(cfg0, scene0):
    grasps = make_grasps(scene0)
    ref, _ = build(cfg0, scene0, **{"profile.scorer_seeds": 64})
    pool = [(g, s) for g in grasps for s in candidates(ref, g)]
    print(f"pool: {len(pool)} stations from {sum(1 for g in grasps if any(gg is g for gg, _ in pool))}/{len(grasps)} grasps "
          f"(ref group {ref.group})", flush=True)
    ref_rob = np.array(score_pool(ref, pool)); del ref
    rows = []
    for seeds in (16, 8, 4):
        for group in (4, 8, 16, 32):
            pl, setup = build(cfg0, scene0, **{"profile.scorer_seeds": seeds, "profile.scorer_group": group})
            score_pool(pl, pool[:pl.group]); sync()                       # warm-up at this batch shape
            t = time.perf_counter(); rob = np.array(score_pool(pl, pool)); sync(); dt = time.perf_counter() - t
            err = rob - ref_rob
            row = {"scorer_seeds": seeds, "group_req": group, "group_used": pl.group, "setup_s": setup,
                   "score_s": dt, "ms_per_station": 1000 * dt / len(pool), "mean_err": float(err.mean()),
                   "mae": float(np.abs(err).mean()), "pass_1.0": int((rob >= 1 - 1e-9).sum()),
                   "ge_0.5": int((rob >= 0.5).sum()), "peak_mb": torch.cuda.max_memory_allocated() / 2**20}
            rows.append(row); del pl
            print(json.dumps(row), flush=True)
    print(f"ref (64 seeds): pass_1.0={int((ref_rob >= 1 - 1e-9).sum())} ge_0.5={int((ref_rob >= 0.5).sum())} "
          f"max={ref_rob.max():.2f}", flush=True)
    print("@@JSON@@" + json.dumps({"pool": len(pool), "rows": rows}))


def sweep_warm(cfg0, scene0):
    grasps = make_grasps(scene0)
    ref, _ = build(cfg0, scene0, **{"profile.scorer_seeds": 64})
    pool = [(g, s) for g in grasps for s in candidates(ref, g)]
    ref_rob = np.array(score_pool(ref, pool)); del ref
    gid = np.array([grasps.index(g) for g, _ in pool])
    best_ref = {int(i): float(ref_rob[gid == i].max()) for i in set(gid)}
    print(f"pool {len(pool)}  ref best per grasp {best_ref}", flush=True)
    for seeds, warm in ((16, False), (8, False), (4, True), (2, True), (1, True)):
        pl, _ = build(cfg0, scene0, **{"profile.scorer_seeds": seeds, "profile.scorer_group": 4})
        score_pool(pl, pool[:pl.group], warm); sync()
        ts, robs = [], []
        for _ in range(3):
            t = time.perf_counter(); rob = np.array(score_pool(pl, pool, warm)); sync(); ts.append(time.perf_counter() - t)
            robs.append(rob)
        rob = np.mean(robs, 0); err = np.concatenate([r - ref_rob for r in robs])
        best = {int(i): round(float(rob[gid == i].max()), 2) for i in set(gid)}
        print(json.dumps({"seeds": seeds, "warm": warm, "score_s": round(min(ts), 3),
                          "ms_per_station": round(1000 * min(ts) / len(pool), 1), "mean_err": round(float(err.mean()), 4),
                          "mae": round(float(np.abs(err).mean()), 4), "max_abs_err": round(float(np.abs(err).max()), 3),
                          "ge_0.5": int((rob >= 0.5).sum()), "ref_ge_0.5": int((ref_rob >= 0.5).sum()),
                          "best_per_grasp": best, "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20)}), flush=True)
        del pl


# ---------------------------------------------------------------- e2e: N grasps, pick one
def pick_and_plan(pl, scored, th, floor):
    """scored: list of (grasp_idx, station, robustness). Best robustness first, nearest on ties; plan the arm for the
    first that works. Same acceptance rules as plan(): >= threshold, else the fallback down to min_accept."""
    order = sorted(scored, key=lambda t: (-t[2], math.hypot(t[1][0], t[1][1])))
    for gi, st, rb in order:
        if rb < floor - 1e-9 or rb <= 0:
            break
        set_grasp(pl, pl._grasps[gi])
        arm = pl.plan_arm(tuple(st[:3]))
        if arm is not None:
            return gi, st, rb
    return None


def e2e(cfg0, scene0, seeds, group, k_max, reps=3, wb=128, stop0=False, skip_a=False):
    grasps = make_grasps(scene0)
    th, floor = cfg0["robustness"]["threshold"], cfg0["robustness"]["min_accept"]
    res = {}
    if skip_a:
        return e2e_b(cfg0, scene0, grasps, th, floor, res, seeds, group, k_max, reps, wb, stop0)
    # A. today's pipeline, once per grasp (whole-body IK, score all, arm plan), then keep the best
    pl, setup = build(cfg0, scene0)
    res["setup_default_s"] = setup
    for _ in range(1):                                                    # warm-up
        set_grasp(pl, grasps[0]); pl.plan()
    tA = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); out = []
        for g in grasps:
            set_grasp(pl, g); out.append(pl.plan())
        sync(); tA.append(time.perf_counter() - t)
    okA = [(i, r) for i, r in enumerate(out) if r["ok"]]
    bestA = max(okA, key=lambda t: t[1]["robustness"]) if okA else None
    res["A_per_grasp_plan"] = {"wall_s": tA, "n_ok": len(okA),
                               "reasons": [r["reason"] for r in out],
                               "rob": [r["robustness"] for r in out],
                               "best": None if not bestA else {"grasp": bestA[0], "rob": bestA[1]["robustness"]}}
    print("A", json.dumps(res["A_per_grasp_plan"]), flush=True)
    del pl
    return e2e_b(cfg0, scene0, grasps, th, floor, res, seeds, group, k_max, reps, wb, stop0)


def e2e_b(cfg0, scene0, grasps, th, floor, res, seeds, group, k_max, reps, wb, stop0):
    # B. score every grasp's nearest k_max stations together (mixed batches), then plan the arm once
    pl, setup = build(cfg0, scene0, **{"profile.scorer_seeds": seeds, "profile.scorer_group": group,
                                       "profile.wb_seeds": wb})
    res["setup_fast_s"] = setup
    pl._grasps = grasps
    candidates(pl, grasps[0]); score_pool(pl, [(grasps[0], np.zeros(9))] * pl.group); sync()
    tB, parts = [], []
    for _ in range(reps):
        sync(); t0 = time.perf_counter()
        pool, t_ik = [], []
        for gi, g in enumerate(grasps):
            ta = time.perf_counter(); S = candidates(pl, g, k_max, stop0); sync(); t_ik.append(round(time.perf_counter() - ta, 3))
            pool += [(gi, s) for s in S]
        sync(); t1 = time.perf_counter()
        rob = score_pool(pl, [(grasps[gi], s) for gi, s in pool]) if pool else []
        sync(); t2 = time.perf_counter()
        pick = pick_and_plan(pl, [(gi, s, r) for (gi, s), r in zip(pool, rob)], th, floor)
        sync(); t3 = time.perf_counter()
        tB.append(t3 - t0); parts.append({"ik_filter_s": t1 - t0, "score_s": t2 - t1, "arm_s": t3 - t2, "pool": len(pool), "ik_per_grasp": t_ik})
    per_g = {}
    for (gi, _), r in zip(pool, rob):
        per_g[gi] = max(per_g.get(gi, 0.0), r)
    res["B_score_all_plan_once"] = {"wb_seeds": wb, "stop_on_zero": stop0, "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
                                    "resident_mb": pl.MEM["resident_after_setup"], "scorer_seeds": seeds, "group_used": pl.group, "k_max": k_max, "wall_s": tB,
                                    "parts": parts, "best_rob_per_grasp": per_g,
                                    "pick": None if not pick else {"grasp": pick[0], "rob": pick[2]}}
    print("B", json.dumps(res["B_score_all_plan_once"]), flush=True)
    res["grasps"] = [{k: g[k] for k in ("yaw_deg", "pitch_deg")} for g in grasps]
    print("@@JSON@@" + json.dumps(res, default=float))


if __name__ == "__main__":
    cfg0, scene0 = load_test_scene()
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        sweep(cfg0, scene0)
    elif mode == "warm":
        sweep_warm(cfg0, scene0)
    else:
        a = dict(arg.split("=") for arg in sys.argv[2:])
        e2e(cfg0, scene0, int(a.get("seeds", 8)), int(a.get("group", 4)), int(a.get("k", 8)) or None,
            wb=int(a.get("wb", 128)), stop0=a.get("stop0") == "1", skip_a=a.get("skip_a") == "1")
