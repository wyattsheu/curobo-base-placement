"""How many obstacle boxes can the IK / scoring never touch? (CPU only; no cuRobo needed)

  python validation/reach_cull.py <maps_root> [site ...]

Every configuration that IK or scoring accepts has the tool at the grasp pose, so:
  - an arm collision sphere on link k is within chain(k → tool) + |centre| + radius of the grasp point
  - a Kachaka body sphere is within (arm reach) + |sphere − arm mount| + radius of it, and never higher than its own top
A box that lies outside both regions cannot collide with any accepted configuration, so dropping it from the IK and
scoring scenes cannot change any result. (Bounds use the box's circumscribed sphere, so they only ever keep more.)
The arm motion planner starts from the stowed arm, so it keeps today's scene; the parking check keeps every box.
"""
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from robot_model import load_config, robot_dict, kachaka_spheres, curobo_yml_path  # noqa: E402


WINDOW = 0.5                    # base search window half-width (m) when the robot is already next to the target


def reach_bounds(cfg):
    """(R_arm, R_body_xy, body_top): grasp-centred bounds for the arm spheres and the Kachaka body spheres."""
    rd = robot_dict(cfg)["kinematics"]
    urdf = ET.parse(rd["urdf_path"]).getroot()
    parent = {}                                                   # child link → (parent link, |joint offset|, travel)
    for j in urdf.findall("joint"):
        o = j.find("origin"); xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        lim = j.find("limit"); travel = 0.0
        if j.get("type") == "prismatic" and lim is not None:
            travel = max(abs(float(lim.get("lower", 0))), abs(float(lim.get("upper", 0))))
        parent[j.find("child").get("link")] = (j.find("parent").get("link"), float(np.linalg.norm(xyz)), travel)

    def chain_to_base(link):                                      # links from `link` down to base_link, with lengths
        out, l = [], link
        while l in parent and l != "base_link":
            p, d, t = parent[l]; out.append((l, d + t)); l = p
        return out

    tool = cfg["robot"]["tool_offset_m"]
    to_gb = {l: d for l, d in chain_to_base("gripper_base")}
    up = chain_to_base("gripper_base")                            # gripper_base … link1
    dist_tool = {"gripper_base": tool}
    acc = tool
    for l, d in up:                                               # each link frame's distance to the tool, along the chain
        acc += d
        p = parent[l][0]
        dist_tool[p] = acc
    dist_tool.setdefault("base_link", acc)
    R_arm = 0.0
    for link, sph in rd["collision_spheres"].items():
        if link == "kachaka_link" or not sph:
            continue
        if link in dist_tool:
            base = dist_tool[link]
        else:                                                     # side branches (fingers): via their parent
            p, d, t = parent[link]; base = dist_tool[p] + d + t
        R_arm = max(R_arm, base + max(np.linalg.norm(s["center"]) + s["radius"] for s in sph))
    mount = np.array(cfg["robot"]["arm_mount_xyz"], float)
    ks = kachaka_spheres(cfg)
    reach = dist_tool["base_link"]                                # arm base (Piper base_link) to tool, at most
    R_body_xy = reach + max(math.hypot(*(np.array(k["center"][:2]) - mount[:2])) + k["radius"] for k in ks)
    body_top = max(k["center"][2] + k["radius"] for k in ks)
    return R_arm, R_body_xy, body_top, reach


def keep_mask(boxes, g, R_arm, R_body_xy, body_top, window=None, reach=None, mount=None, body_r=None):
    """window = half-width (m) of the base search window around the current pose (base frame origin), or None.
    With a window the base centre stays inside it, so the Kachaka spheres stay within window + body_r, and the arm
    spheres within window + |mount| + reach of the origin as well."""
    C = np.array([b["center"] for b in boxes], float); D = np.array([b["dims"] for b in boxes], float)
    half = np.linalg.norm(D, axis=1) / 2; half_xy = np.linalg.norm(D[:, :2], axis=1) / 2
    near_arm = np.linalg.norm(C - g, axis=1) - half <= R_arm
    near_body = (C[:, 2] - D[:, 2] / 2 <= body_top) & (np.linalg.norm(C[:, :2] - g[:2], axis=1) - half_xy <= R_body_xy)
    if window is not None:
        w = window * math.sqrt(2)                                   # corner of the square window
        near_arm &= np.linalg.norm(C[:, :2], axis=1) - half_xy <= w + np.linalg.norm(mount[:2]) + reach
        near_body &= np.linalg.norm(C[:, :2], axis=1) - half_xy <= w + body_r
    return near_arm | near_body


def main(root, sites):
    import glob, os
    from bench_lab_maps import load_site, start_pose
    import scene_io
    cfg = load_config(HERE.parent / "config.yaml")
    R_arm, R_xy, top, reach = reach_bounds(cfg)
    print(f"bounds: arm spheres within {R_arm:.3f} m of the grasp; Kachaka spheres within {R_xy:.3f} m (horizontal) "
          f"and below {top:.3f} m; arm base to tool <= {reach:.3f} m", flush=True)
    sites = sites or sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "*"))
                            if glob.glob(os.path.join(p, "*_sem_color_fine.npz")))
    mount = np.array(cfg["robot"]["arm_mount_xyz"], float)
    body_r = max(math.hypot(*k["center"][:2]) + k["radius"] for k in kachaka_spheres(cfg))
    tot = [0, 0, 0]
    for site in sites:
        try:
            data, objs, occ = load_site(os.path.join(root, site))
        except (IndexError, FileNotFoundError, ValueError) as e:
            print(f"skip {site}: {e}"); continue
        for o in objs:
            st = start_pose(occ, np.array(o["pos"][:2]))
            if st is None:
                continue
            sc = scene_io.scene_from_slam(data, st, o["pos"], [1.0, 0.0, 0.0, 0.0], cfg, target=o["target"], occupancy=occ)
            g = np.array(sc.grasp_pos_base, float)
            k = keep_mask(sc.boxes_base, g, R_arm, R_xy, top)
            kw = keep_mask(sc.boxes_base, g, R_arm, R_xy, top, WINDOW, reach, mount, body_r)
            tot[0] += len(k); tot[1] += int(k.sum()); tot[2] += int(kw.sum())
            print(f"{site}/{o['cls']}#{o['inst']}: {len(k)} boxes → reach {int(k.sum())} ({100 * k.mean():.0f}%)"
                  f" → reach + window ±{WINDOW} m {int(kw.sum())} ({100 * kw.mean():.0f}%)", flush=True)
    print(f"total {tot[0]} → reach {tot[1]} ({100 * tot[1] / max(tot[0], 1):.0f}%) → + window {tot[2]} "
          f"({100 * tot[2] / max(tot[0], 1):.0f}%)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
