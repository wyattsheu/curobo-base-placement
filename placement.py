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

from robot_model import robot_cfg, in_safety_cone, footprint_xy, kachaka_spheres, n_collision_spheres, arm_reach, BASE_JOINTS, ARM_JOINTS

MPPI_PARTICLES = 25          # particles per seed in cuRobo v2's IK MPPI stage (sets the kernel size above)
NEAR_MARGIN_M = 0.05         # beyond the arm's reach: cuRobo's largest collision activation distance (0.01 m) + slack
IK_POS_TOL_M = 0.05          # a solved gripper pose is at least this close to the goal (cuRobo's threshold is mm-level)
from scene_io import base_to_map_pose, base_to_map_xy


def _sync():
    torch.cuda.synchronize()


def _qz(t):
    return np.stack([np.cos(t / 2), np.zeros_like(t), np.zeros_like(t), np.sin(t / 2)], -1)


def _qmul(a, b):                                              # wxyz, broadcast over leading dims
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], -1)


class BasePlacementPlanner:
    def __init__(self, cfg, scene=None):
        """With `scene`: solvers sized for that one request. Without: a long-running planner with fixed capacities
        (profile.max_scene_boxes / max_near_boxes); every request then comes in through set_scene()."""
        self.cfg, self.scene = cfg, None
        P, Rb, S = cfg["profile"], cfg["robustness"], cfg["search"]
        self.T, self.MEM = {}, {}
        t0 = time.perf_counter()
        d, a = Rb["perturb_xy_m"], math.radians(Rb["perturb_yaw_deg"])
        self.d = d
        self.OFFS = np.array([(dx, dy, dp) for dx in (-d, 0, d) for dy in (-d, 0, d) for dp in (-a, 0, a)])
        # every robot sphere, in any arm pose, stays within `reach` of the shoulder (`pivot`): a box farther away never comes
        # within the activation distance of any sphere, so it adds exactly nothing to any cost or collision check
        self.pivot, reach, self.tool_reach = arm_reach(cfg)
        self.reach = reach + NEAR_MARGIN_M if P.get("near_boxes_only", True) else np.inf
        if scene is not None:                                   # one request: capacities fitted to this scene
            self._take_scene(scene)
            n_scene, n_near = max(P["collision_cache_cuboids"], len(self.names) + 2), self._near_bound(d) + 2
            window = self.window
        else:                                                   # long-running: fixed capacities
            self._take_scene(None)
            n_scene, n_near = P["max_scene_boxes"], P["max_near_boxes"]
            window = {"x": list(S["window_x"]), "y": list(S["window_y"])}
        cache, near_cache = {"cuboid": n_scene}, {"cuboid": n_near}  # whole-body IK (base moves anywhere) / one station
        self.T["scene_cache"], self.T["near_cache"] = float(n_scene), float(n_near)
        # one warp kernel covers envs x seeds x particles x robot spheres x cuboids and must stay below 2^31 work items:
        # with many boxes (real maps) or many spheres, score fewer stations per call instead of failing
        per_station = len(self.OFFS) * P["scorer_seeds"] * MPPI_PARTICLES * n_collision_spheres(cfg) * near_cache["cuboid"]
        self.group = max(1, min(P["scorer_group"], (2**31 - 1) // per_station))
        self.T["scorer_group_used"] = float(self.group)
        rc_wb, rc_arm = robot_cfg(cfg, False, window), robot_cfg(cfg, True)
        empty = {"cuboid": {"_": {"dims": [0.01] * 3, "pose": [0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0]}}}   # filled per request
        # 1. whole-body IK: one problem (the decided gripper pose)
        self.ik_wb = InverseKinematics(InverseKinematicsCfg.create(robot=rc_wb, scene_model=empty,
                     collision_cache=cache, num_seeds=P["wb_seeds"], self_collision_check=True))
        self.cub_wb = self._cuboids(self.ik_wb)
        # 3. robustness: one problem per (candidate, perturbation), each with its own copy of the scene
        self.E = self.group * len(self.OFFS)
        self.ik_rob = InverseKinematics(InverseKinematicsCfg.create(robot=rc_arm, scene_model=[empty] * self.E,
                     collision_cache=near_cache, num_seeds=P["scorer_seeds"], self_collision_check=True, max_batch_size=self.E, multi_env=True))
        self.ik_rob.config.exit_early = False                   # solve_envs checks the LM seeds itself (see there)
        self.cub = self._cuboids(self.ik_rob)
        # 4. arm-only motion planner
        self.planner = MotionPlanner(MotionPlannerCfg.create(robot=rc_arm, scene_model=empty,
                     collision_cache=near_cache, use_cuda_graph=True,
                     num_ik_seeds=P["planner_ik_seeds"], num_trajopt_seeds=P["planner_trajopt_seeds"]))
        self.planner.warmup(enable_graph=True, num_warmup_iterations=1)
        self.base_sph = np.array([[*k['center'], k['radius']] for k in kachaka_spheres(cfg)])   # 2. parking check: the base only
        st = self.planner.default_joint_state
        stow = cfg["robot"].get("stow_arm")
        self.stow = np.array(stow if stow else [float(st.position.reshape(-1)[list(st.joint_names).index(j)]) for j in ARM_JOINTS])
        start = st.clone(); flat = start.position.view(-1); names = list(start.joint_names)
        for j, v in zip(ARM_JOINTS, self.stow):                 # the arm trajectory starts from the stowed arm
            flat[names.index(j)] = float(v)
        self.start = start.unsqueeze(0)
        self.write_boxes(self.cub_wb, np.zeros((1, 3)), near=False)
        for _ in range(2):                                      # warm-up (CUDA graphs)
            self.ik_wb.solve_pose(self.goal_from(np.zeros((1, 3))), return_seeds=P["wb_seeds"])
        self.robustness(np.zeros((self.group, 3)))
        if scene is not None:
            self.set_scene(scene)
        _sync(); self.T["setup_s"] = time.perf_counter() - t0
        self.MEM["resident_after_setup"] = torch.cuda.memory_allocated() / 2**20
        self.MEM["setup_peak"] = torch.cuda.max_memory_allocated() / 2**20

    @staticmethod
    def _cuboids(solver):
        d_ = solver.scene_collision_checker.data
        return getattr(d_, "cuboid", None) or getattr(d_, "cuboids")

    def _take_scene(self, scene):
        """Gripper pose, search window and obstacle arrays (base frame) of one request; no GPU work. Without a scene:
        nothing to avoid and a gripper pose in front of the robot, enough for the warm-up."""
        self.scene = scene
        self.gpos = np.array(scene.grasp_pos_base if scene else (0.5, 0.0, 0.7), float)   # decided gripper pose — fixed
        self.gquat = np.array(scene.grasp_quat_base if scene else (1.0, 0.0, 0.0, 0.0), float)
        S = self.cfg["search"]
        self.window = self._window() if scene else {"x": list(S["window_x"]), "y": list(S["window_y"])}
        # only the boxes some base pose in the search window (± navigation error) can touch
        B = [b for b in (scene.boxes_base if scene else []) if self._window_gap(b) < self.reach + math.hypot(*self.pivot[:2]) + self.d]
        self.names = [b["name"] for b in B]
        self.C = np.array([b["center"] for b in B], float).reshape(-1, 3); self.D = np.array([b["dims"] for b in B], float).reshape(-1, 3)
        self.Y = np.array([b.get("yaw", 0.0) for b in B], float)
        self.T["boxes_used"] = float(len(B))

    def set_scene(self, scene):
        """Next request: its obstacles go straight into the solvers' collision tensors; nothing is rebuilt."""
        t = time.perf_counter()
        self._take_scene(scene)
        if len(self.names) > self.cub_wb.max_n:
            raise ValueError(f"{len(self.names)} boxes in the search window, profile.max_scene_boxes is {self.cub_wb.max_n}")
        # a loose upper bound (any yaw, 5 cm grid); the exact count per station is checked in write_boxes(), which refuses
        # rather than drop a box, so this is only reported
        self.T["near_bound"] = float(self._near_bound(self.d))
        self.write_boxes(self.cub_wb, np.zeros((1, 3)), near=False)
        self.T["set_scene_s"] = time.perf_counter() - t

    # ------------------------------------------------------------ geometry helpers
    def _window(self):
        S = self.cfg["search"]
        wx, wy = list(S["window_x"]), list(S["window_y"])
        x0, x1, y0, y1 = self.scene.known.bbox_base()           # never search outside the area the SLAM has seen
        return {"x": [max(wx[0], x0), min(wx[1], x1)], "y": [max(wy[0], y0), min(wy[1], y1)]}

    def _window_gap(self, b):
        """Lower bound of the horizontal distance from box b to the search window (≤ 0 when they overlap)."""
        (x0, x1), (y0, y1) = self.window["x"], self.window["y"]
        cx, cy = b["center"][:2]
        return math.hypot(max(x0 - cx, 0, cx - x1), max(y0 - cy, 0, cy - y1)) - math.hypot(*b["dims"][:2]) / 2

    def _box_dist(self, p):
        """Distance from points p (M, 3) to every box → (M, n_boxes); negative inside."""
        c, s = np.cos(self.Y), np.sin(self.Y)
        dx, dy = p[:, None, 0] - self.C[None, :, 0], p[:, None, 1] - self.C[None, :, 1]
        q = np.abs(np.stack([c * dx + s * dy, -s * dx + c * dy, p[:, None, 2] - self.C[None, :, 2]], -1)) - self.D / 2
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)

    def _near(self, F):
        """(len(F), n_boxes): which boxes the robot can touch, in any arm pose, with its base standing at each row of F."""
        F = np.atleast_2d(F); c, s = np.cos(F[:, 2]), np.sin(F[:, 2]); px, py, pz = self.pivot
        return self._box_dist(np.c_[F[:, 0] + c * px - s * py, F[:, 1] + s * px + c * py, np.full(len(F), pz)]) < self.reach

    def _near_bound(self, d):
        """Most boxes one station can touch (sizes the collision cache; never decides which boxes count): a 5 cm grid over
        the base poses robustness() and plan_arm() can see — stations (the gripper reaches the target, so the base is
        within tool_reach + |pivot_xy| of it) and the warm-up at the origin, ± navigation error d. Each grid point counts
        the boxes within reach of any shoulder position around it (the shoulder is |pivot_xy| from the base centre,
        a base at most h/√2 from its nearest grid point)."""
        if not np.isfinite(self.reach) or not len(self.names):
            return len(self.names)
        h = 0.05; (x0, x1), (y0, y1) = self.window["x"], self.window["y"]; pxy = math.hypot(*self.pivot[:2])
        G = np.stack(np.meshgrid(np.arange(min(x0, 0) - d, max(x1, 0) + d + h, h),
                                 np.arange(min(y0, 0) - d, max(y1, 0) + d + h, h)), -1).reshape(-1, 2)
        slack = d * math.sqrt(2) + h
        G = G[((np.hypot(*(G - self.gpos[:2]).T) <= self.tool_reach + IK_POS_TOL_M + pxy + slack) & self._parkable_near(G, slack))
              | (np.hypot(*G.T) <= slack)]
        r = self.reach + pxy + h / math.sqrt(2)
        return max(int((self._box_dist(np.c_[g, np.full(len(g), self.pivot[2])]) < r).sum(1).max())
                   for g in np.array_split(G, max(1, len(G) // 256)))

    def _parkable_near(self, G, slack):
        """Base positions G (M, 2) close enough to a drivable cell that a station within `slack` of them could park:
        parking_ok() needs the body centre (|body_center_xy| from the base origin) on a known cell."""
        K = self.scene.known
        if getattr(K, "_edt", None) is None:
            from scipy.ndimage import distance_transform_edt
            K._edt = distance_transform_edt(~K.known) * K.resolution            # distance to the nearest known cell
        ij = np.floor((base_to_map_xy(G, K.base_pose_in_map) - K.origin_xy) / K.resolution + 0.5).astype(int)
        H, W = K.known.shape
        inside = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
        dist = np.full(len(G), np.inf); dist[inside] = K._edt[ij[inside, 1], ij[inside, 0]]
        body = math.hypot(*self.cfg["robot"]["kachaka"].get("body_center_xy", [0.0, 0.0]))
        return dist <= body + slack + K.resolution * math.sqrt(2)

    def _in_frame(self, f):
        """Obstacle centres / yaws expressed in the frame of a base standing at f = (x, y, yaw)."""
        bx, by, bt = f; c, s = math.cos(-bt), math.sin(-bt)
        dx, dy = self.C[:, 0] - bx, self.C[:, 1] - by
        return np.c_[c * dx - s * dy, s * dx + c * dy, self.C[:, 2]], self.Y - bt

    def scene_at(self, f, near=False):
        cen, yaw = self._in_frame(f); keep = self._near(f)[0] if near else np.ones(len(self.names), bool)
        return Scene(cuboid=[Cuboid(name=n, dims=list(d), pose=[*c, math.cos(y / 2), 0, 0, math.sin(y / 2)])
                             for n, c, d, y, k in zip(self.names, cen, self.D, yaw, keep) if k])

    def scene_dict(self, f, near=False):
        cen, yaw = self._in_frame(f); keep = self._near(f)[0] if near else np.ones(len(self.names), bool)
        return {"cuboid": {n: {"dims": list(map(float, d)), "pose": [*map(float, c), math.cos(y / 2), 0, 0, math.sin(y / 2)]}
                           for n, c, d, y, k in zip(self.names, cen, self.D, yaw, keep) if k}}

    def load_envs(self, envf):
        """Obstacles of every robustness environment (base standing at envf[e] = (x, y, yaw)): only the boxes within reach."""
        self.write_boxes(self.cub, envf)

    def write_boxes(self, cub, envf, near=True):
        """Obstacles of every environment of a solver (base standing at envf[e] = (x, y, yaw)) written straight into its
        collision tensors, packed to the front, inverse pose in closed form. near: only the boxes within reach."""
        near = self._near(envf) if near else np.ones((len(envf), len(self.names)), bool); n = near.sum(1)
        if n.max(initial=0) > cub.max_n:
            raise RuntimeError(f"{n.max()} boxes for one environment, the collision cache holds {cub.max_n}")
        idx = np.argsort(~near, axis=1, kind="stable")[:, :cub.max_n]; K = idx.shape[1]   # kept boxes first, scene order
        # obstacle-from-base pose for every env: R = Rz(psi - yaw_o), t = Rz(-yaw_o) (b - c_o)
        C, Y = self.C[idx], self.Y[idx]
        cy, sy = np.cos(-Y), np.sin(-Y)
        dx = envf[:, None, 0] - C[..., 0]; dy = envf[:, None, 1] - C[..., 1]
        inv = np.zeros(idx.shape + (7,), np.float32)
        inv[..., 0] = cy * dx - sy * dy; inv[..., 1] = sy * dx + cy * dy; inv[..., 2] = -C[..., 2]
        ang = envf[:, None, 2] - Y
        inv[..., 3] = np.cos(ang / 2); inv[..., 6] = np.sin(ang / 2)
        dev = cub.inv_pose.device
        cub.inv_pose[:, :K, :7] = torch.from_numpy(inv).to(dev)
        cub.dims[:, :K, :3] = torch.from_numpy(self.D[idx].astype(np.float32)).to(dev)
        cub.enable[:, :K] = torch.from_numpy((np.arange(K)[None] < n[:, None]).astype(np.uint8)).to(dev)
        cub.count[:] = torch.from_numpy(n.astype(np.int32)).to(dev)

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
        q = q[ok][:, [jn.index(j) for j in BASE_JOINTS + ARM_JOINTS]]
        (x0, x1), (y0, y1) = self.window["x"], self.window["y"]  # a long-running planner's joint limits are the config
        return q[(q[:, 0] >= x0) & (q[:, 0] <= x1) & (q[:, 1] >= y0) & (q[:, 1] <= y1)]   # window; this request's is smaller

    # ------------------------------------------------------------ step 2
    def parking_ok(self, bases):
        """Kachaka parked at each pose: its base spheres clear of every box, and its footprint on drivable cells.
        Only the base counts here (where it can go); the arm and anything above the base are checked by the arm planner."""
        bases = np.atleast_2d(bases)[:, :3]
        c0, s0 = np.cos(bases[:, 2])[:, None], np.sin(bases[:, 2])[:, None]
        Sx = bases[:, :1] + c0 * self.base_sph[None, :, 0] - s0 * self.base_sph[None, :, 1]
        Sy = bases[:, 1:2] + s0 * self.base_sph[None, :, 0] + c0 * self.base_sph[None, :, 1]
        S = np.stack([Sx, Sy, np.broadcast_to(self.base_sph[None, :, 2], Sx.shape), np.broadcast_to(self.base_sph[None, :, 3], Sx.shape)], -1)
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
        target stays in the safety cone). All (candidate, perturbation) problems go into ONE multi-env IK call; each
        environment only holds the boxes within reach of its base (load_envs)."""
        bases = np.atleast_2d(bases)[:, :3]; n = len(bases)
        fr = (bases[:, None, :] + self.OFFS[None]).reshape(-1, 3)
        envf = np.vstack([fr, np.repeat(fr[-1:], self.E - len(fr), 0)]) if len(fr) < self.E else fr
        self.load_envs(envf)
        cone = in_safety_cone(self.cfg, envf, self.gpos[:2])               # outside it a perturbation fails anyway
        ok = self.solve_envs(self.goal_from(envf), care=cone)[:len(fr)]
        return (ok & cone[:len(fr)]).reshape(n, len(self.OFFS)).mean(axis=1)

    def solve_envs(self, goal, warm=None, care=None):
        """Can the arm reach `goal` in each robustness environment? (bool per env)
        cuRobo's IK first runs Levenberg-Marquardt on random starts (pose only), then MPPI + L-BFGS with collisions from
        those seeds. Measured on real maps, that second stage turns many LM seeds that were ALREADY valid (collision-free,
        within limits, on the pose) into failures — more so with more seeds and later in the batch — while it rescues
        others. Both are checked by the same metrics, so an env counts as reachable if either the LM seeds or the
        optimised seeds contain a valid solution; the optimiser only runs when some env is still unsolved.
        warm (E, dof): a known arm solution per env, put in place of the first seed.
        care (E,) bool: envs whose answer matters (e.g. inside the safety cone); the others never trigger the optimiser."""
        ik, ns = self.ik_rob, self.cfg["profile"]["scorer_seeds"]
        g = goal.reorder_links(ik.kinematics.tool_frames)
        seeds = ik.seed_ik_solver.solve_batch(goal_tool_poses=g, return_seeds=ns).js_solution.position.view(self.E, ns, -1).clone()
        if warm is not None:
            seeds[:, 0] = warm
        lm = ik.config.use_lm_seed; ik.config.use_lm_seed = False            # seeds given: do not redo LM
        try:
            ok = ik.solve_pose(goal, seed_config=seeds, return_seeds=ns, run_optimizer=False).success.reshape(self.E, -1).any(dim=1)
            need = ~ok if care is None else ~ok & torch.as_tensor(care, device=ok.device)
            if bool(need.any()):
                ok |= ik.solve_pose(goal, seed_config=seeds, return_seeds=ns).success.reshape(self.E, -1).any(dim=1)
        finally:
            ik.config.use_lm_seed = lm
        return ok.cpu().numpy()

    # ------------------------------------------------------------ step 4
    def plan_arm(self, base):
        self.planner.update_world(self.scene_at(base, near=True))
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
        P, Rb = self.cfg["profile"], self.cfg["robustness"]
        th, floor = Rb["threshold"], Rb.get("min_accept", 0.0)
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
            done = {t["idx"] for t in tried}
            for j in sorted(checked, key=lambda j: (-checked[j], dist[j])):
                if checked[j] <= 0 or checked[j] < floor - 1e-9:
                    break
                if j in done:
                    continue
                arm = self._timed("arm_plan_s", self.plan_arm, tuple(U[j, :3]))
                tried.append({"idx": j, "ok": arm is not None, "robustness": checked[j]})
                if arm is not None:
                    best, below = j, True; break
        # why there is no answer, at the first step that ran out of candidates
        if best is not None:
            reason = "ok"
        elif len(Q) == 0:
            reason = "no_ik_solution"        # no 9-DoF IK solution anywhere in the search window
        elif n_cone == 0:
            reason = "outside_cone"          # solutions exist, but none keeps the target in the arm's ±30° safety cone
        elif not park.any():
            reason = "cannot_park"           # the base collides or leaves the drivable area at every candidate
        elif not tried:
            reason = "not_robust"            # parkable, but every candidate scores below min_accept
        else:
            reason = "arm_plan_failed"       # robust enough, but no arm trajectory from the stowed pose
        res = {"ok": best is not None, "reason": reason, "below_threshold": below,
               "robustness": checked.get(best) if best is not None else None,
               "max_robustness": max(checked.values()) if checked else None,
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
