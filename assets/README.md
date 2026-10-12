# assets/

Robot129 = AgileX Piper 6-DoF arm (+ gripper) as used in the lab's simulation hand-off
(`robot129_pro6000_sim_20260913`). Copied here unchanged, except two paths in `robot129.yml`:

| file | from the hand-off | change |
|---|---|---|
| `robot129.yml` | `research/configs/curobo/robot129.yml` | `urdf_path`, `asset_root_path` made relative to this folder (`robot_model.py` turns them into absolute paths) |
| `robot129_description/urdf/robot129.urdf` | `ros2_ws/src/robot129_description/urdf/robot129.urdf` | none |
| `robot129_description/meshes/*.STL` | `ros2_ws/src/robot129_description/meshes/` (from `piper_description`) | none; only the 10 meshes the URDF uses |

cuRobo plans with the collision spheres in `robot129.yml` (`load_meshes: false`); the STL meshes are only for drawing.
