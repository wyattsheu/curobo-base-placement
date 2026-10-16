"""The lab's exported 3D semantic map (robotic_agent `demo/<site>/` or `robot_ws/data/lab/<site>/`) → this project's input.

The export is NOT the SLAM_DOCKER_HANDOFF §4 npz: it is a set of files that object_query reads together:

    robot_deploy_<site>_sem_color_fine.npz           points (N, 3) in the 3D map frame (y vertical), colors (N, 3)
    <site>_instance_labels.npz                       centers (M, 3) voxel centres, cluster_id (M,) instance per voxel
    robot_deploy_<site>_instance_semantic_table_fine.json   labels [class names], voxel_size, instances [{inst_id, class_id, ...}]
    alignment_<site>.yaml                            3D map → Kachaka 2D map (plane_fit + sim2)

This module rebuilds what scene_io expects: means3D in the KACHAKA MAP frame (metres, z up, floor at z = 0),
semantic_ids (class id per point, 255 = no instance), instance_ids (-1 = none), and the class-name table.
The 3D → map transform is the one object_query_server.transform_points_to_map uses:

    d = p - mu;  uv = (d·e1, d·e2);  xy = s R uv + t;  z = s (d·n)       (+ a constant so the floor is at 0)

    python lab_map.py <export_dir> --out map.npz --targets 13 19 24     # adds target_inst13, ... masks
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml


def _one(d, pattern):
    hits = sorted(glob.glob(os.path.join(d, pattern)))
    if not hits:
        raise FileNotFoundError(f"{pattern} not found in {d}")
    return hits[0]


def load_alignment(path):
    a = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pf, s2 = a["plane_fit"], a["sim2"]
    return {"mu": np.array([pf["mu"]["x"], pf["mu"]["y"], pf["mu"]["z"]], float),
            "e1": np.array(pf["basis_e1"], float), "e2": np.array(pf["basis_e2"], float), "n": np.array(pf["normal_n"], float),
            "s": float(s2["s"]), "R": np.array(s2["R"], float), "t": np.array([s2["t"]["x"], s2["t"]["y"]], float)}


def map3d_to_map(P, A):
    """3D map frame → Kachaka map frame (xy) + height along the plane normal, both scaled to metres."""
    d = np.asarray(P, float) - A["mu"]
    uv = np.c_[d @ A["e1"], d @ A["e2"]]
    return A["s"] * uv @ A["R"].T + A["t"], A["s"] * (d @ A["n"])


def floor_height(h, sem, floor_ids):
    """Floor level: the densest height among floor-labelled points (falls back to the lowest dense layer)."""
    hh = h[np.isin(sem, floor_ids)] if len(floor_ids) and np.isin(sem, floor_ids).sum() > 1000 else h[h < np.percentile(h, 20)]
    hist, edges = np.histogram(hh, bins=np.arange(hh.min(), hh.max() + 0.02, 0.01))
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


def load_lab_map(export_dir, voxel=0.02):
    pts = np.load(_one(export_dir, "*_sem_color_fine.npz"))["points"].astype(float)
    lab = np.load(_one(export_dir, "*_instance_labels.npz"))
    table = json.loads(Path(_one(export_dir, "*_instance_semantic_table_fine.json")).read_text(encoding="utf-8"))
    A = load_alignment(_one(export_dir, "alignment_*.yaml"))
    from scipy.spatial import cKDTree
    vs = float(table.get("voxel_size", 0.05))
    dist, k = cKDTree(lab["centers"]).query(pts, distance_upper_bound=vs * 0.9)         # point → its instance voxel
    inst = np.full(len(pts), -1, np.int32); hit = np.isfinite(dist); inst[hit] = lab["cluster_id"][k[hit]]
    cls_of = {int(i["inst_id"]): int(i["class_id"]) for i in table["instances"] if not i.get("filtered", False)}
    inst[~np.isin(inst, list(cls_of))] = -1
    sem = np.full(len(pts), 255, np.uint8)
    m = inst >= 0; sem[m] = np.vectorize(cls_of.get)(inst[m])
    names = {i: n for i, n in enumerate(table["labels"])}
    xy, h = map3d_to_map(pts, A)
    floor = floor_height(h, sem, [i for i, n in names.items() if n == "floor"])
    P = np.c_[xy, h - floor]
    if voxel:                                                                           # thin to one point per voxel
        _, keep = np.unique(np.floor(P / voxel).astype(np.int64), axis=0, return_index=True)
        keep = np.sort(keep); P, sem, inst = P[keep], sem[keep], inst[keep]
    # floor_height_m: the height subtracted above. Frames that keep the raw plane-normal height (Grasp Lab Global's
    # "native metric map") are this frame shifted up by it.
    return {"means3D": P.astype(np.float32), "semantic_ids": sem, "instance_ids": inst.astype(np.int32),
            "floor_height_m": np.float32(floor)}, names, table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir")
    ap.add_argument("--out", required=True, help="output .npz (a class-name .json is written next to it)")
    ap.add_argument("--targets", type=int, nargs="*", default=[], help="instance ids to add as target_inst<id> masks")
    ap.add_argument("--voxel", type=float, default=0.02)
    a = ap.parse_args(argv)
    data, names, table = load_lab_map(a.export_dir, a.voxel)
    for t in a.targets:
        data[f"target_inst{t}"] = data["instance_ids"] == t
    np.savez_compressed(a.out, **data)
    Path(a.out).with_suffix(".json").write_text(json.dumps({str(k): v for k, v in names.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    P = data["means3D"]
    print(f"{a.out}: {len(P)} points, x {P[:, 0].min():.2f}..{P[:, 0].max():.2f}, y {P[:, 1].min():.2f}..{P[:, 1].max():.2f}, "
          f"z {np.percentile(P[:, 2], 1):.2f}..{np.percentile(P[:, 2], 99):.2f}; instances {len(np.unique(data['instance_ids'])) - 1}")
    for t in a.targets:
        q = P[data[f"target_inst{t}"]]
        cls = next((i["class_name"] for i in table["instances"] if i["inst_id"] == t), "?")
        print(f"  target {t} ({cls}): {len(q)} points, median ({np.median(q[:, 0]):.3f}, {np.median(q[:, 1]):.3f}, {np.median(q[:, 2]):.3f})"
              f", z {q[:, 2].min():.2f}..{q[:, 2].max():.2f}")


if __name__ == "__main__":
    main()
