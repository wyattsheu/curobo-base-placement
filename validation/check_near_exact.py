"""Is dropping the far boxes exact? cuRobo's own FK on many random arm poses:
  1. every collision sphere stays inside the reach bound (|c - pivot| + r <= reach)
  2. for real stations on a real map, no sphere ever comes within the activation distance (0.01 m) of a dropped box,
     i.e. each dropped box adds exactly 0 to cuRobo's collision cost (penetration = r + eta - sdf <= 0).
  python validation/check_near_exact.py <maps_root> <site> <ab_json_log>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/validation")
from robot_model import load_config, robot_cfg, arm_reach  # noqa: E402
import scene_io  # noqa: E402
from bench_lab_maps import load_site, start_pose  # noqa: E402
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg  # noqa: E402
from curobo.types import JointState  # noqa: E402

root, site, ablog = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = load_config(REPO + "/config.yaml")
pivot, reach = arm_reach(cfg)
ETA = 0.01
rc = robot_cfg(cfg, True)
ik = InverseKinematics(InverseKinematicsCfg.create(robot=rc, num_seeds=1, self_collision_check=False))
kc = rc.kinematics.kinematics_config
names = list(ik.joint_names)
lim = kc.joint_limits.position.cpu().numpy()
print("active joints", names, "limits", np.round(lim, 2).tolist())

torch.manual_seed(0)
dev = torch.device("cuda")
piv = torch.tensor(pivot, device=dev, dtype=torch.float32)
worst = 0.0; S_all = []
for mode in ("limits", "any angle"):
    for _ in range(20):                                  # 20 x 10k poses per mode
        n = 10000
        lo, hi = torch.tensor(lim[0], device=dev), torch.tensor(lim[1], device=dev)
        q = lo + (hi - lo) * torch.rand(n, len(names), device=dev)
        if mode == "any angle":                          # the bound must hold for ANY joint values
            q = (torch.rand(n, len(names), device=dev) * 2 - 1) * math.pi
        ks = ik.compute_kinematics(JointState.from_position(q.float(), joint_names=names))
        sph = ks.robot_spheres.reshape(n, -1, 4)
        far = (torch.linalg.norm(sph[..., :3] - piv, dim=-1) + sph[..., 3])
        far = far[sph[..., 3] > 0]
        worst = max(worst, float(far.max()))
        if mode == "limits" and len(S_all) < 2:
            S_all.append(sph[:2000])
    print(f"[{mode}] farthest sphere surface from the shoulder: {worst:.4f} m   (bound {reach:.4f} m)", flush=True)
assert worst <= reach + 1e-5

# 2. the A/B run's stations on the real map: dropped boxes vs the arm spheres in every perturbed env frame
row = json.loads(next(l for l in open(ablog) if l.startswith("@@JSON@@"))[8:])
data, objs, occ = load_site(f"{root}/{site}")
o = objs[0]; start = start_pose(occ, np.array(o["pos"][:2]))
scene = scene_io.scene_from_slam(data, start, o["pos"], [1.0, 0.0, 0.0, 0.0], cfg, target=o["target"], occupancy=occ)
B = scene.boxes_base
C = torch.tensor([b["center"] for b in B], device=dev, dtype=torch.float32)
D = torch.tensor([b["dims"] for b in B], device=dev, dtype=torch.float32)
Y = torch.tensor([b.get("yaw", 0.0) for b in B], device=dev, dtype=torch.float32)
d, a = cfg["robustness"]["perturb_xy_m"], math.radians(cfg["robustness"]["perturb_yaw_deg"])
OFFS = np.array([(dx, dy, dp) for dx in (-d, 0, d) for dy in (-d, 0, d) for dp in (-a, 0, a)])
S = torch.cat(S_all).reshape(-1, 4)                      # 4000 random poses x all spheres (arm-model frame)
S = S[S[:, 3] > 0]
R_ = reach + 0.05                                        # placement.NEAR_MARGIN_M
min_gap, n_drop, n_env = np.inf, [], 0
for gi, st in row["pool"]:
    for off in OFFS:
        x, y, t = np.array(st[:3]) + off
        c, s = math.cos(t), math.sin(t)
        P = torch.stack([x + c * S[:, 0] - s * S[:, 1], y + s * S[:, 0] + c * S[:, 1], S[:, 2]], 1)   # spheres → base frame
        pv = torch.tensor([x + c * pivot[0] - s * pivot[1], y + s * pivot[0] + c * pivot[1], pivot[2]], device=dev)
        def sdf(p):                                      # (M, 3) → (M, n_boxes)
            cy, sy = torch.cos(Y), torch.sin(Y)
            dx, dy = p[:, None, 0] - C[None, :, 0], p[:, None, 1] - C[None, :, 1]
            qq = torch.stack([cy * dx + sy * dy, -sy * dx + cy * dy, p[:, None, 2] - C[None, :, 2]], -1).abs() - D / 2
            return torch.linalg.norm(qq.clamp(min=0), dim=-1) + qq.max(-1).values.clamp(max=0)
        drop = sdf(pv[None])[0] >= R_                    # what placement.load_envs leaves out
        n_drop.append(int(drop.sum())); n_env += 1
        if drop.any():
            Cd, Dd, Yd = C[drop], D[drop], Y[drop]
            C_, D_, Y_ = C, D, Y; C, D, Y = Cd, Dd, Yd
            for k in range(0, len(P), 4096):
                g = (sdf(P[k:k + 4096]) - S[k:k + 4096, 3:4]).min()
                min_gap = min(min_gap, float(g))
            C, D, Y = C_, D_, Y_
print(f"{n_env} env frames ({len(row['pool'])} stations x 27), dropped boxes per env: median {np.median(n_drop):.0f} "
      f"of {len(B)}; {len(S)} sphere positions each", flush=True)
print(f"closest any sphere surface came to a dropped box: {min_gap:.4f} m  (cuRobo counts a box only below {ETA} m)", flush=True)
assert min_gap > ETA
print("EXACT: every dropped box adds exactly 0 to the collision cost")
