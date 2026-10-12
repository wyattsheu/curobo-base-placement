"""Build the 9-DoF cuRobo robot (Kachaka x/y/yaw + Piper 6 axes) from config.yaml.

Kinematic chain (cuRobo v2 `extra_links`, no URDF edits):

    world (= base frame, Kachaka now)
      └─ base_x   X_PRISM  ┐
          └─ base_y   Y_PRISM  │ virtual joints = where Kachaka should go, relative to now
              └─ base_yaw Z_ROT ┘  (link "kachaka_link" carries the Kachaka collision spheres)
                  └─ arm_mount FIXED  ← config robot.arm_mount_xyz / arm_mount_rpy_deg
                      └─ base_link … joint1..6 … gripper_base … pinch_center   (robot129.urdf)
"""
import math
from pathlib import Path

import numpy as np
import yaml

BASE_JOINTS = ["base_x", "base_y", "base_yaw"]
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name("config.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_dir"] = str(path.resolve().parent)                  # relative paths in the config are relative to it
    return cfg


def curobo_yml_path(cfg):
    p = Path(cfg["robot"]["curobo_yml"])
    if "asset_root" in cfg["robot"]:                          # old style: absolute asset folder + path inside it
        p = Path(cfg["robot"]["asset_root"]) / p
    return p if p.is_absolute() else Path(cfg.get("_dir", Path(__file__).parent)) / p


def quat_from_rpy(r, p, y):
    """wxyz quaternion from roll/pitch/yaw (radians)."""
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def kachaka_spheres(cfg):
    """Collision spheres of the Kachaka body (+ extra_spheres), in the base frame."""
    k = cfg["robot"]["kachaka"]
    L, W, _ = k["size_m"]; r, z = k["sphere_radius"], k["sphere_z"]; nx, ny = k["sphere_grid"]
    cx, cy = k.get("body_center_xy", [0.0, 0.0])                    # body centre relative to the turning axis
    xs = cx + np.linspace(-L / 2 + r, L / 2 - r, nx); ys = cy + np.linspace(-W / 2 + r, W / 2 - r, ny)
    out = [{"center": [float(x), float(y), z], "radius": r} for x in xs for y in ys]
    out += [{"center": [float(a), float(b), float(c)], "radius": float(rr)} for a, b, c, rr in (k.get("extra_spheres") or [])]
    return out


def arm_mount(cfg):
    """(xyz, wxyz) of arm_base in the base frame."""
    xyz = cfg["robot"]["arm_mount_xyz"]
    rpy = [math.radians(v) for v in cfg["robot"]["arm_mount_rpy_deg"]]
    return list(xyz), quat_from_rpy(*rpy)


def robot_dict(cfg, window=None):
    """cuRobo robot dict: robot129.yml + virtual base joints + Kachaka spheres + measured mount."""
    R = cfg["robot"]; S = cfg["search"]
    yml = curobo_yml_path(cfg)
    d = yaml.safe_load(yml.read_text(encoding="utf-8"))
    k, cs = d["kinematics"], d["kinematics"]["cspace"]
    tool = (k.get("extra_links") or {}).get(R["tool_frame"])
    if tool is not None and R.get("tool_offset_m") is not None:     # where the decided gripper pose sits on the gripper
        tool["fixed_transform"][2] = float(R["tool_offset_m"])
    for key in ("urdf_path", "asset_root_path"):              # relative to the yml → absolute (cuRobo would resolve
        if k.get(key) and not Path(k[key]).is_absolute():     # a relative path against its own content folder)
            k[key] = str((yml.parent / k[key]).resolve())
    wx, wy = (window or {}).get("x", S["window_x"]), (window or {}).get("y", S["window_y"])

    def joint(name, parent, link, jtype, lim, vel):
        return {"parent_link_name": parent, "link_name": link, "joint_name": name, "joint_type": jtype,
                "fixed_transform": [0, 0, 0, 1, 0, 0, 0], "joint_limits": list(lim), "joint_velocity_limits": vel}

    xyz, q = arm_mount(cfg)
    k["extra_links"] = {**(k.get("extra_links") or {}),
        "base_x_link": joint("base_x", "world", "base_x_link", "X_PRISM", wx, [-0.5, 0.5]),
        "base_y_link": joint("base_y", "base_x_link", "base_y_link", "Y_PRISM", wy, [-0.5, 0.5]),
        "kachaka_link": joint("base_yaw", "base_y_link", "kachaka_link", "Z_ROT", [-S["yaw_limit_rad"], S["yaw_limit_rad"]], [-1.0, 1.0]),
        "arm_mount": {"parent_link_name": "kachaka_link", "child_link_name": "base_link", "link_name": "arm_mount",
                      "joint_name": "kachaka_to_arm", "joint_type": "FIXED", "fixed_transform": [*xyz, *q]}}
    k["collision_spheres"]["kachaka_link"] = kachaka_spheres(cfg)
    k["collision_link_names"] = list(k["collision_link_names"]) + ["kachaka_link"]
    k["mesh_link_names"] = list(k["collision_link_names"])
    k["self_collision_ignore"]["kachaka_link"] = ["base_link", "link1", "arm_mount"]
    k["self_collision_ignore"]["base_link"] = list(k["self_collision_ignore"]["base_link"]) + ["kachaka_link"]
    k["self_collision_buffer"]["kachaka_link"] = 0.0
    n = len(BASE_JOINTS)
    cs["joint_names"] = BASE_JOINTS + list(cs["joint_names"])
    cs["default_joint_position"] = [0.0] * n + list(cs["default_joint_position"])
    cs["cspace_distance_weight"] = [1.0, 1.0, 0.5] + list(cs["cspace_distance_weight"])
    for key in ("acceleration_scale", "jerk_scale", "velocity_scale", "null_space_maximum_distance", "null_space_weight"):
        cs[key] = [1.0] * n + list(cs[key])
    cs["max_acceleration"] = [2.0] * n + list(cs["max_acceleration"])
    cs["max_jerk"] = [50.0] * n + list(cs["max_jerk"])
    return d


def robot_cfg(cfg, lock_base=False, window=None):
    """cuRobo RobotCfg; lock_base=True gives the arm-only (6-DoF) model with Kachaka fixed at the origin."""
    from curobo._src.types.robot import RobotCfg
    from curobo._src.types.device_cfg import DeviceCfg
    d = robot_dict(cfg, window)
    if lock_base:
        d["kinematics"]["lock_joints"].update({j: 0.0 for j in BASE_JOINTS})
    rc = RobotCfg.create(d, DeviceCfg())
    kc = rc.kinematics.kinematics_config
    i = list(kc.joint_names).index("joint1")
    lim = cfg["robot"]["joint1_limit_rad"]
    kc.joint_limits.position[0, i] = -lim
    kc.joint_limits.position[1, i] = lim
    return rc


def mount_xy_in_base(cfg, b):
    """Arm-base position (x, y) in the base frame when Kachaka stands at b = (x, y, yaw); vectorised over b."""
    b = np.atleast_2d(b)
    mx, my = cfg["robot"]["arm_mount_xyz"][:2]
    c, s = np.cos(b[:, 2]), np.sin(b[:, 2])
    return b[:, 0] + c * mx - s * my, b[:, 1] + s * mx + c * my


def in_safety_cone(cfg, b, target_xy):
    """base_action.SAFE_BEARING_RAD: the target must lie within ±bearing of the arm's forward direction."""
    b = np.atleast_2d(b)
    mx, my = mount_xy_in_base(cfg, b)
    arm_yaw = b[:, 2] + math.radians(cfg["robot"]["arm_mount_rpy_deg"][2])
    a = np.arctan2(target_xy[1] - my, target_xy[0] - mx) - arm_yaw
    return np.abs((a + np.pi) % (2 * np.pi) - np.pi) <= cfg["robot"]["bearing_limit_rad"]


def footprint_xy(cfg, nx=3, ny=3):
    """Sample points of the Kachaka body outline in the base frame (body centre is ahead of the turning axis)."""
    k = cfg["robot"]["kachaka"]; L, W = k["size_m"][:2]; cx, cy = k.get("body_center_xy", [0.0, 0.0])
    return np.array([[cx + x, cy + y] for x in np.linspace(-L / 2, L / 2, nx) for y in np.linspace(-W / 2, W / 2, ny)])
