"""Where should Kachaka go so that the arm can reach the ALREADY-DECIDED gripper pose? (plan B, cuRobo v2)

    planner = BasePlacementPlanner(cfg, scene)      # builds every solver once (≈ 6 s, do it at start-up)
    result  = planner.plan()                        # base pose to drive to + arm check at that pose

The gripper pose (position + orientation of tool_frame) comes from upstream and is never changed here.
Steps inside plan():
  1. whole_body()   9-DoF IK (Kachaka x, y, yaw + 6 arm joints) for that one gripper pose → many candidate base poses
  2. parking_ok()   Kachaka + stowed arm parked there: no collision, and standing on cells the SLAM has seen
  3. robustness()   nearest candidate first: lock the base at 27 perturbed poses (navigation error) and re-solve the
                    6-DoF arm IK for the same gripper pose, all in ONE multi-env IK call; keep the first that passes
  4. plan_arm()     base locked at that pose: 6-DoF MotionPlanner from the stowed arm to the gripper pose;
                    on failure go back to 3 with the next candidate
Everything is in the base frame (Kachaka now). result["chosen_map"] is the goal for Navigate_to_pose.
"""
import math
import time

import numpy as np
import torch

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import GoalToolPose, Pose

from robot_model import robot_cfg, in_safety_cone, footprint_xy, BASE_JOINTS, ARM_JOINTS
from scene_io import base_to_map_pose


def _sync():
    torch.cuda.synchronize()


def _qz(t):
    return np.stack([np.cos(t / 2), np.zeros_like(t), np.zeros_like(t), np.sin(t / 2)], -1)


def _qmul(a, b):                                              # wxyz, broadcast over leading dims
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


class BasePlacementPlanner:
    def __init__(self, cfg, scene):
        self.cfg, self.scene = cfg, scene
        P, Rb = cfg["profile"], cfg["robustness"]
        self.T, self.MEM = {}, {}
        t0 = time.perf_counter()
        self.gpos = np.array(scene.grasp_pos_base, float)       # decided gripper pose (base frame) — fixed
        self.gquat = np.array(scene.grasp_quat_base, float)
        self.window = self._window()
        B = scene.boxes_base                                    # obstacles as arrays (base frame)
        self.names = [b["name"] for b in B]
        self.C = np.array([b["center"] for b in B], float); self.D = np.array([b["dims"] for b in B], float)
        self.Y = np.array([b.get("yaw", 0.0) for b in B], float)
        cache = {"cuboid": max(P["collision_cache_cuboids"], len(B) + 2)}
        d, a = Rb["perturb_xy_m"], math.radians(Rb["perturb_yaw_deg"])
        self.OFFS = np.array([(dx, dy, dp) for dx in (-d, 0, d) for dy in (-d, 0, d) for dp in (-a, 0, a)])
        self.group = P["scorer_group"]
        rc_wb, rc_arm = robot_cfg(cfg, False, self.window), robot_cfg(cfg, True)
        # 1. whole-body IK: one problem (the decided gripper pose)
        self.ik_wb = InverseKinematics(InverseKinematicsCfg.create(robot=rc_wb, scene_model=self.scene_at((0, 0, 0)),
                     collision_cache=cache, num_seeds=P["wb_seeds"], self_collision_check=True))
        # 3. robustness: one problem per (candidate, perturbation), each with its own copy of the scene
        self.E = self.group * len(self.OFFS)
        self.ik_rob = InverseKinematics(InverseKinematicsCfg.create(robot=rc_arm, scene_model=[self.scene_dict((0, 0, 0))] * self.E,
                     collision_cache=cache, num_seeds=P["scorer_seeds"], self_collision_check=True, max_batch_size=self.E, multi_env=True))
        d_ = self.ik_rob.scene_collision_checker.data
        self.cub = getattr(d_, "cuboid", None) or getattr(d_, "cuboids")
        # 4. arm-only motion planner
        self.planner = MotionPlanner(MotionPlannerCfg.create(robot=rc_arm, scene_model=self.scene_at((0, 0, 0)),
                     collision_cache=cache, use_cuda_graph=False,
                     num_ik_seeds=P["planner_ik_seeds"], num_trajopt_seeds=P["planner_trajopt_seeds"]))
        self.planner.warmup(enable_graph=True, num_warmup_iterations=1)
        # 2. batched forward kinematics for the parking check (only get_kinematics is used: validate() is broken in this release)
        from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
        self.fk = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(robot_config=rc_wb, scene_model=self.scene_at((0, 0, 0)),
                  n_cuboids=cache["cuboid"], n_meshes=0, collision_activation_distance=0.0))
        st = self.planner.default_joint_state
        stow = cfg["robot"].get("stow_arm")
        self.stow = np.array(stow if stow else [float(st.position.reshape(-1)[list(st.joint_names).index(j)]) for j in ARM_JOINTS])
        start = st.clone(); flat = start.position.view(-1); names = list(start.joint_names)
        for j, v in zip(ARM_JOINTS, self.stow):                 # the arm trajectory starts from the stowed arm
            flat[names.index(j)] = float(v)
        self.start = start.unsqueeze(0)
        for _ in range(2):                                      # warm-up (CUDA graphs)
            self.ik_wb.solve_pose(self.goal_from(np.zeros((1, 3))), return_seeds=P["wb_seeds"])
        self.robustness(np.zeros((self.group, 3)))
        _sync(); self.T["setup_s"] = time.perf_counter() - t0
        self.MEM["resident_after_setup"] = torch.cuda.memory_allocated() / 2**20
        self.MEM["setup_peak"] = torch.cuda.max_memory_allocated() / 2**20

    # ------------------------------------------------------------ geometry helpers
    def _window(self):
        S = self.cfg["search"]
        wx, wy = list(S["window_x"]), list(S["window_y"])
        x0, x1, y0, y1 = self.scene.known.bbox_base()           # never search outside the area the SLAM has seen
        return {"x": [max(wx[0], x0), min(wx[1], x1)], "y": [max(wy[0], y0), min(wy[1], y1)]}

    def _in_frame(self, f):
        """Obstacle centres / yaws expressed in the frame of a base standing at f = (x, y, yaw)."""
        bx, by, bt = f; c, s = math.cos(-bt), math.sin(-bt)
        dx, dy = self.C[:, 0] - bx, self.C[:, 1] - by
        return np.c_[c * dx - s * dy, s * dx + c * dy, self.C[:, 2]], self.Y - bt

    def scene_at(self, f):
        cen, yaw = self._in_frame(f)
        return Scene(cuboid=[Cuboid(name=n, dims=list(d), pose=[*c, math.cos(y / 2), 0, 0, math.sin(y / 2)])
                             for n, c, d, y in zip(self.names, cen, self.D, yaw)])

    def scene_dict(self, f):
        cen, yaw = self._in_frame(f)
        return {"cuboid": {n: {"dims": list(map(float, d)), "pose": [*map(float, c), math.cos(y / 2), 0, 0, math.sin(y / 2)]}
                           for n, c, d, y in zip(self.names, cen, self.D, yaw)}}

    def goal_from(self, F):
        """The decided gripper pose, expressed in the frame of a base standing at each row of F = [(x, y, yaw)]."""
        F = np.atleast_2d(F)
        c, s = np.cos(-F[:, 2]), np.sin(-F[:, 2]); px, py = self.gpos[0] - F[:, 0], self.gpos[1] - F[:, 1]
        pos = np.stack([c * px - s * py, s * px + c * py, np.full(len(F), self.gpos[2])], 1)
        quat = _qmul(_qz(-F[:, 2]), np.broadcast_to(self.gquat, (len(F), 4)))
        return GoalToolPose.from_poses({self.cfg["robot"]["tool_frame"]: Pose(
            position=torch.tensor(pos, device="cuda", dtype=torch.float32),
            quaternion=torch.tensor(quat, device="cuda", dtype=torch.float32))}, num_goalset=1)

    # ------------------------------------------------------------ step 1
    def whole_body(self):
        ns = self.cfg["profile"]["wb_seeds"]
        r = self.ik_wb.solve_pose(self.goal_from(np.zeros((1, 3))), return_seeds=ns)
        jn = list(r.js_solution.joint_names)
        q = r.js_solution.position.reshape(-1, len(jn)).cpu().numpy()
        ok = r.success.reshape(-1).cpu().numpy().astype(bool)
        return q[ok][:, [jn.index(j) for j in BASE_JOINTS + ARM_JOINTS]]

    # ------------------------------------------------------------ step 2
    def parking_ok(self, bases):
        """Kachaka + stowed arm parked at each pose: collision-free, and all of the footprint on known cells.
        cuRobo gives every collision sphere (batched FK); the scene is boxes, so the distance is exact."""
        bases = np.atleast_2d(bases)[:, :3]
        q = np.c_[bases, np.tile(self.stow, (len(bases), 1))]
        kin = self.fk.get_kinematics(torch.tensor(q, device="cuda", dtype=torch.float32)[:, None, :])
        sph = kin.robot_spheres.reshape(len(q), -1, 4).cpu().numpy()
        S = sph[:, sph[0, :, 3] > 0]                                         # [n, spheres, 4]
        p = S[:, :, None, :3] - self.C[None, None]                           # → each box frame (rotate by -yaw)
        c, s = np.cos(-self.Y)[None, None], np.sin(-self.Y)[None, None]
        loc = np.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1], p[..., 2]], -1)
        d = np.abs(loc) - self.D[None, None] / 2
        sdf = np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0) - S[:, :, None, 3]
        ok = sdf.min(axis=(1, 2)) > 0
        fp = footprint_xy(self.cfg)
        for n, b in enumerate(bases):
            if ok[n]:
                R = np.array([[math.cos(b[2]), -math.sin(b[2])], [math.sin(b[2]), math.cos(b[2])]])
                ok[n] = self.scene.known.is_known_base(fp @ R.T + b[:2]).all()
        return ok

    # ------------------------------------------------------------ step 3
    def robustness(self, bases):
        """Fraction of the 27 perturbed base poses from which the arm still reaches the decided gripper pose (and the
        target stays in the safety cone). All (candidate, perturbation) problems go into ONE multi-env IK call; every
        environment's obstacles are written straight into the collision tensor (closed-form inverse pose)."""
        bases = np.atleast_2d(bases)[:, :3]; n = len(bases)
        fr = (bases[:, None, :] + self.OFFS[None]).reshape(-1, 3)
        envf = np.vstack([fr, np.repeat(fr[-1:], self.E - len(fr), 0)]) if len(fr) < self.E else fr
        # obstacle-from-base pose for every env: R = Rz(psi - yaw_o), t = Rz(-yaw_o) (b - c_o)
        cy, sy = np.cos(-self.Y)[None], np.sin(-self.Y)[None]
        dx = envf[:, None, 0] - self.C[None, :, 0]; dy = envf[:, None, 1] - self.C[None, :, 1]
        inv = np.zeros((self.E, len(self.names), 7), np.float32)
        inv[:, :, 0] = cy * dx - sy * dy; inv[:, :, 1] = sy * dx + cy * dy; inv[:, :, 2] = -self.C[None, :, 2]
        ang = envf[:, None, 2] - self.Y[None]
        inv[:, :, 3] = np.cos(ang / 2); inv[:, :, 6] = np.sin(ang / 2)
        self.cub.inv_pose[:, :len(self.names), :7] = torch.from_numpy(inv).to(self.cub.inv_pose.device)
        r = self.ik_rob.solve_pose(self.goal_from(envf))
        ok = r.success.reshape(self.E, -1).any(dim=1).cpu().numpy()[:len(fr)]
        return (ok & in_safety_cone(self.cfg, fr, self.gpos[:2])).reshape(n, len(self.OFFS)).mean(axis=1)

    # ------------------------------------------------------------ step 4
    def plan_arm(self, base):
        self.planner.update_world(self.scene_at(base))
        pr = self.planner.plan_pose(self.goal_from(np.asarray([base], float)), self.start, max_attempts=5)
        if pr is None or not bool(pr.success.any()):
            return None
        n = int(pr.interpolated_last_tstep.view(-1)[0]); names = list(pr.interpolated_trajectory.joint_names)
        tr = pr.interpolated_trajectory.position.reshape(-1, len(names))[:n].cpu().numpy()
        return {"traj": tr[:, [names.index(j) for j in ARM_JOINTS]].tolist(), "dt": float(pr.interpolated_trajectory.dt.view(-1)[0])}

    # ------------------------------------------------------------ pipeline
    def _timed(self, name, fn, *a):
        _sync(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated(); t = time.perf_counter()
        out = fn(*a)
        _sync(); self.T[name] = self.T.get(name, 0.0) + time.perf_counter() - t
        self.MEM[name] = max(self.MEM.get(name, 0.0), (torch.cuda.max_memory_allocated() - m0) / 2**20)
        self.MEM["online_peak"] = max(self.MEM.get("online_peak", 0.0), torch.cuda.max_memory_allocated() / 2**20)
        return out

    def _candidates(self, Q):
        cone = in_safety_cone(self.cfg, Q[:, :3], self.gpos[:2]) if len(Q) else np.zeros(0, bool)
        cand = Q[cone]
        key = np.round(np.c_[cand[:, :2] * 100, np.degrees(cand[:, 2]) / 2])          # 1 cm, 2 deg
        _, u = np.unique(key, axis=0, return_index=True); u = np.sort(u)
        park = self.parking_ok(cand[u]) if len(u) else np.zeros(0, bool)
        return int(cone.sum()), cand[u], park

    def plan(self):
        P, th = self.cfg["profile"], self.cfg["robustness"]["threshold"]
        for k in ("wholebody_ik_s", "filter_s", "scoring_s", "arm_plan_s"):
            self.T[k] = 0.0
        # 1.+2. whole-body IK (fresh seeds each round) until enough parkable candidates
        Q, rounds = np.zeros((0, 9)), 0
        while rounds < P["wb_rounds_max"]:
            Q = np.vstack([Q, self._timed("wholebody_ik_s", self.whole_body)]); rounds += 1
            n_cone, U, park = self._timed("filter_s", self._candidates, Q)
            if park.sum() >= P["min_candidates"]:
                break
        dist = np.hypot(U[:, 0], U[:, 1]) if len(U) else np.zeros(0)
        order = [int(j) for j in np.argsort(dist) if park[j]]
        # 3.+4. nearest first: robustness, then plan the first one that passes
        checked, tried, best, arm = {}, [], None, None
        for g0 in range(0, len(order), self.group):
            idx = np.array(order[g0:g0 + self.group])
            rob = self._timed("scoring_s", self.robustness, U[idx, :3])
            checked.update(zip(idx.tolist(), rob.tolist()))
            for j, rj in zip(idx.tolist(), rob.tolist()):
                if rj < th - 1e-9:
                    continue
                arm = self._timed("arm_plan_s", self.plan_arm, tuple(U[j, :3]))
                tried.append({"idx": j, "ok": arm is not None, "robustness": rj})
                if arm is not None:
                    best = j; break
            if best is not None:
                break
        below = False
        if best is None:                     # nobody reached the threshold: try the most robust ones, and say so
            for j in sorted(checked, key=lambda j: (-checked[j], dist[j])):
                if checked[j] <= 0:
                    break
                arm = self._timed("arm_plan_s", self.plan_arm, tuple(U[j, :3]))
                tried.append({"idx": j, "ok": arm is not None, "robustness": checked[j]})
                if arm is not None:
                    best, below = j, True; break
        res = {"ok": best is not None, "below_threshold": below, "robustness": checked.get(best) if best is not None else None,
               "n_success": int(len(Q)), "wb_rounds": rounds, "n_cone": n_cone, "window": self.window,
               "q_success": Q.tolist(), "uniq": U.tolist(), "parked_ok": park.tolist(),
               "rob": [checked.get(i) for i in range(len(U))], "n_checked": len(checked), "tried_plans": tried}
        if best is not None:
            ch = tuple(float(v) for v in U[best, :3])
            res.update({"chosen_base": ch, "chosen_map": base_to_map_pose(ch, self.scene.base_pose_in_map),
                        "chosen_whole_body_q": U[best].tolist(), "arm": arm,
                        "rank_by_distance": int(np.argsort(dist).tolist().index(best)) + 1})
        res["timing_s"] = dict(self.T); res["memory_mb"] = dict(self.MEM)
        return res
