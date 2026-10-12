"""Pack curobo_base_placement (code + config + inputs) into ONE script that can be piped to a remote Python:

    python bundle.py | ssh <server> 'cd /tmp && <cuRobo env>/bin/python -'

The remote side unpacks into a RAM-only temporary directory (/dev/shm), runs run_demo.main(), and the directory is
deleted when the run ends — nothing is installed or left behind. On the robot (Thor) you do not need this: just
`python run_demo.py ...` inside the folder.
Every extra argument is passed to run_demo.py; files named by --npz / --request / --config are packed too.
`--entry other.py` runs that script's main(args) instead of run_demo (e.g. ../experiments/slam_scope_bench.py).
"""
import base64
import sys
from pathlib import Path

HERE = Path(__file__).parent
CODE = ["robot_model.py", "scene_io.py", "placement.py", "run_demo.py", "lab_map.py", "config.yaml",
        "assets/robot129.yml", "assets/robot129_description/urdf/robot129.urdf"]   # STL meshes are not needed by cuRobo (load_meshes: false)
FILE_ARGS = {"--config", "--npz", "--request"}
LAB_MAP_FILES = ["*_sem_color_fine.npz", "*_instance_labels.npz", "*_instance_semantic_table_fine.json", "alignment_*.yaml",
                 "occupancy_*.yaml", "occupancy_*.png"]
DEFAULT_INPUTS = ["test_maps/slam.npz", "test_maps/slam.json", "test_maps/request.json"]


def main():
    argv = sys.argv[1:]
    files = {name: (HERE / name).read_bytes() for name in CODE}
    for name in DEFAULT_INPUTS:
        if (HERE / name).exists():
            files[name] = (HERE / name).read_bytes()
    out_args, entry = [], "run_demo"
    if "--entry" in argv:
        k = argv.index("--entry"); src = Path(argv[k + 1]); argv = argv[:k] + argv[k + 2:]
        files["_entry.py"] = src.read_bytes(); entry = "_entry"
    i = 0
    while i < len(argv):                                                      # pack user-given input files
        a = argv[i]; out_args.append(a)
        if a == "--lab-map" and i + 1 < len(argv):                           # the lab's export: only the 4 files used
            d = Path(argv[i + 1])
            for pat in LAB_MAP_FILES:
                for f in sorted(d.glob(pat))[:1]:
                    files[f"inputs/lab_map/{f.name}"] = f.read_bytes()
            out_args.append("inputs/lab_map"); i += 2; continue
        if a in FILE_ARGS and i + 1 < len(argv):
            p = Path(argv[i + 1]); key = f"inputs/{p.name}"
            files[key] = p.read_bytes(); out_args.append(key)
            if p.suffix == ".npz" and p.with_suffix(".json").exists():
                files[f"inputs/{p.with_suffix('.json').name}"] = p.with_suffix(".json").read_bytes()
            i += 2; continue
        i += 1
    payload = {k: base64.b64encode(v).decode() for k, v in files.items()}
    sys.stdout.write(f"""import base64, os, sys, tempfile
FILES = {payload!r}
ARGS = {out_args!r}
with tempfile.TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as d:
    for rel, b64 in FILES.items():
        p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(base64.b64decode(b64))
    sys.path.insert(0, d); os.chdir(d)
    import {entry} as _m
    _m.main(ARGS)
""")


if __name__ == "__main__":
    main()
