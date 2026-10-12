# mobile_grasp：被擋住時，Kachaka 要移到哪裡才抓得到（cuRobo v2）

上游已經決定好**抓什麼**和**夾爪最終位姿**（位置＋朝向），但原地被障礙物擋住抓不到。這個專案只做一件事：

> 根據 **3D SLAM 點雲**和**上游請求**，決定 **Kachaka 要停到哪裡**（給 `Navigate_to_pose` 的目標），並確認停過去之後，
> 用**同一個夾爪位姿**手臂抓得到、而且整段動作不會撞到東西。

不做：選抓取方向、改夾爪姿勢、2D 地圖、導航路徑本身（交給 Kachaka 的 `Navigate_to_pose`）。

機器人：Kachaka（移動底盤）＋ AgileX Piper 6 軸手臂（Robot129）。規劃：cuRobo v2 的 9 軸 whole-body IK（底盤 x, y, ψ ＋手臂 6 軸）。

---

## 1. 輸入

### 1.1 3D SLAM 點雲：`.npz`

| 欄位 | 形狀 / 型別 | 必填 | 用途 |
|---|---|---|---|
| `means3D` | (N, 3) float32，公尺 | **必填** | 點的位置，**SLAM 地圖座標**（和 Kachaka 定位用的是同一張地圖）。z 朝上、地板在 z ≈ 0 |
| `semantic_ids` | (N,) uint8 | 選填 | 每個點的類別 id。有的話先依類別分物體；沒有也能用（見 1.4） |
| 目標物遮罩（名稱自訂，例 `target_mask`）| (N,) bool | 建議 | SLAM 標出的**目標物的點**；這些點是要抓的東西，不當障礙物 |
| `logit_opacities` | (N, 1) | 選填 | 3D Gaussian 的不透明度；> 0 的點才採用 |
| 同名 `.json`（例 `slam.json`）| `{"0": "floor", "1": "table", …}` | 選填 | 類別 id → 名稱；讓 request 能用名稱指定類別 |

**不需要**：障礙物方塊、障礙物清單、網格地圖。障礙物由程式從點雲自己算（`scene_io.py:190 objects_to_boxes`）：
地板以上、夾爪位置 1.8 m 內、不是目標物的點，全部都算障礙物。

其他格式（PLY、PCD、ROS PointCloud2）只要轉成 `means3D` 存成 npz 即可，例如：

```python
np.savez("map.npz", means3D=xyz.astype(np.float32))                                   # 最少
np.savez("map.npz", means3D=xyz.astype(np.float32), target_mask=is_target)            # 建議
```

### 1.2 上游請求：`request.json`

```json
{"base_pose_in_map": [2.0, 1.0, 0.5236],
 "grasp_pose_in_map": {"position": [2.797, 1.46, 0.46],
                       "quaternion_wxyz": [0.354, -0.612, 0.354, 0.612]},
 "target": {"mask_key": "target_mask"}}
```

| 欄位 | 意義 |
|---|---|
| `base_pose_in_map` | Kachaka **現在**的定位 (x, y, yaw)，地圖座標，yaw 是弧度 |
| `grasp_pose_in_map.position` | 決定好的夾爪位置（`pinch_center`：兩指中間），地圖座標，公尺 |
| `grasp_pose_in_map.quaternion_wxyz` | 決定好的夾爪朝向，順序 w, x, y, z。`pinch_center` 的 +z 是夾爪前進（接近）方向 |
| `target`（選填）| 哪些點是目標物，三選一，依優先順序：<br>`{"mask_key": "target_mask"}` → npz 裡那個 bool 欄位（**最準**）<br>`{"class": "can"}` 或 `{"class": 5}` → 這個類別、且在夾爪位置 0.2 m 內的點（同類別但較遠的仍是障礙）<br>不給 → 退路：夾爪位置為中心、半徑 7 cm、上下各 10 cm 的直立圓柱內的點 |

### 1.3 為什麼要知道哪些點是目標物

目標物本身也在點雲裡，而夾爪位姿就在它中間。如果不排除它，它會被包成障礙物，手臂一到夾爪位姿就「撞到」，IK 永遠無解。
只給夾爪位置時用圓柱退路排除：直立的小物體（罐子、瓶子）可以；物體比圓柱大、或旁邊 7 cm 內有別的東西時不可靠，請給遮罩或類別。

### 1.4 只有點雲（沒有語意分類）也可以

物體互相接觸時（擋板立在桌上、櫃子靠著桌子），沒有分類就會連成一個物體。程式會再依**高度分層**：
俯視圖每一格取它自己的頂面高度，頂面高度不同的區域（桌面 0.41 m、擋板 0.71 m、櫃子 0.91 m）各自包方塊、往上堆疊，
不會把整片桌面包到櫃子的高度（`scene_io.py:169 height_levels`）。

在測試場景驗證過（RTX PRO 6000）：

| 輸入 | 障礙方塊 | IK 解 | 選中站位 (map) | 軌跡對照有語意的方塊 |
|---|---|---|---|---|
| 點雲＋語意＋目標遮罩 | 10 | 56 | (3.19, 0.86, 117°) | 最小間隙 4.4 mm |
| 點雲＋目標遮罩 | 16 | 56 | 同上 | 4.4 mm |
| 只有點雲（不給目標） | 16 | 56 | 同上 | 4.4 mm |

---

## 2. 輸出

`run_demo.py` 印出一行 `@@JSON@@{…}`：

| 欄位 | 意義 |
|---|---|
| `ok` | 找到站位，而且停過去後手臂規劃成功 |
| **`chosen_map`** | **(x, y, yaw) 地圖座標 → 直接給 Kachaka `Navigate_to_pose`** |
| `chosen_base` | 同一個站位，相對於現在位置（base 座標）的移動量 |
| **`arm.traj`、`arm.dt`** | 停好後，手臂從收起姿態到夾爪位姿的軌跡：N × 6（joint1–6，弧度），每步 `dt` 秒 |
| `robustness` | 假設 Kachaka 停歪 ±5 cm、±5°（27 種），手臂還抓得到的比例 |
| `below_threshold` | true = 沒有站位 27 種全過，改用最穩健的那個 |
| `timing_s`、`memory_mb` | 每一步耗時（秒）與 GPU 記憶體（MB） |
| `n_boxes`、`boxes_base` | 從點雲算出的障礙方塊（base 座標） |
| `target_points`、`target_source` | 排除了幾個目標物的點、用哪種方式（`slam_mask` / `class_near_grasp` / `radius_fallback`）|
| `q_success`、`uniq`、`parked_ok`、`rob`、`tried_plans` | 中間結果（候選站位、停車檢查、評分），除錯與可視化用 |

實機上，Kachaka 到位後應該用**實際到達的位置**重新呼叫一次：`scene_from_slam(…, 到達位置, 同一個夾爪位姿)` ＋只有手臂 6 軸的
MotionPlanner（`robot_cfg(cfg, lock_base=True)`）。`placement.py:178 plan_arm` 就是這個流程。

---

## 3. 怎麼跑

### 有 cuRobo + GPU 的機器（Thor、工作站）

```bash
python run_demo.py                                                   # test_maps/slam.npz + test_maps/request.json
python run_demo.py --npz /maps/xxx.npz --request request.json
```

### 從沒有 GPU 的電腦丟到 server 跑（server 上不留任何檔案）

```bash
python bundle.py | ssh <server> 'cd /tmp && <cuRobo 環境的 python> -'
python bundle.py --npz /maps/xxx.npz --request request.json | ssh <server> '…'
```

`bundle.py` 把程式、config、`assets/` 的 yml 與 URDF、輸入檔打包成一支腳本，在 server 的 `/dev/shm` 解開執行，跑完自動刪除。

### 相依

- cuRobo v2（[NVlabs/curobo](https://github.com/NVlabs/curobo)，commit `4ea7736` "cuRoboV2 research release"，0.8.0）＋ PyTorch（CUDA）
- `requirements.txt`：numpy、scipy、PyYAML
- 機器人資產已在 `assets/`（不用連到外部路徑）

---

## 4. 檔案

```
mobile_grasp/
├─ config.yaml        ← 所有假設值與參數（量到實際數值只改這裡）
├─ run_demo.py        ← 入口：讀點雲與請求 → 規劃 → 印出 JSON
├─ scene_io.py        ← 點雲 + 請求 → 規劃器要的格式（障礙方塊、已知區域、base 座標的夾爪位姿）
├─ robot_model.py     ← 9 軸 cuRobo 模型（Kachaka x, y, ψ + Piper 6 軸）
├─ placement.py       ← 找站位並確認抓得到
├─ bundle.py          ← 打包成一支腳本丟到 server 跑
├─ make_test_maps.py  ← 從 test_scene.yaml 做出測試用的點雲與請求
├─ test_scene.yaml、test_maps/   ← 測試場景（桌上罐子前有擋板、左邊櫃子、後方與右方是牆）
└─ assets/            ← robot129.yml（cuRobo 設定、碰撞球）、robot129_description/（URDF、STL）
```

## 5. 資料流（檔案:行號）

```
slam.npz + request.json（map 座標）
   │ scene_io.py:258 scene_from_slam()
   │   :274 目標物點（遮罩 → 類別 → 圓柱退路，:297）   :288 已知區域   :306 障礙點
   │   :190 objects_to_boxes()  依類別 → 3D 連通物體 → :169 依高度分層 → :124 decompose() 包方塊
   ▼
PlanningScene（base 座標 = Kachaka 現在的位置）
   │ placement.py:46 BasePlacementPlanner(cfg, scene)   一次建好所有 solver（約 9 s，開機時做一次）
   │ placement.py:204 plan()
   │   1 :126 whole_body()   9 軸 IK：同一個夾爪位姿 → 很多候選站位 (x, y, ψ)
   │   2 :196 _candidates()  ±30° 護欄（robot_model.py:124）＋ :135 parking_ok() 停車不撞、在已知區域
   │   3 :158 robustness()   由近到遠，27 種到位誤差各重解 6 軸 IK
   │   4 :178 plan_arm()     底盤鎖住，6 軸 MotionPlanner；失敗就換下一個（:233 都沒達標時用最穩健的）
   ▼
chosen_map → Navigate_to_pose；arm.traj → 到位後的手臂
```

| 座標系 | 定義 | 轉換 |
|---|---|---|
| map | SLAM／Kachaka 地圖 | 輸入檔案都在這裡 |
| base | Kachaka **現在**的位置：底盤中心投影到地面，x 朝車頭、z 朝上 | `scene_io.py:32 map_to_base_xy`、`:49 quat_map_to_base`、`:43 base_to_map_pose`（輸出給導航）|

## 6. 量到實際數值要改哪裡（`config.yaml`）

| 量到的東西 | 欄位 | 進到程式的哪裡 |
|---|---|---|
| 手臂裝在 Kachaka 的位置 | `robot.arm_mount_xyz`、`arm_mount_rpy_deg` | `robot_model.py:82`（運動鏈）、`:124`（護欄）|
| Kachaka 實際尺寸 | `robot.kachaka.size_m` | 碰撞球、停車檢查 |
| 相機桿等凸出物 | `robot.kachaka.extra_spheres` | 碰撞球 |
| 開車時手臂收起的姿態 | `robot.stow_arm`（null = robot129.yml 預設，夾爪在車頭前約 0.5 m）| 停車檢查、手臂規劃起點 |
| Kachaka 到位誤差 | `robustness.perturb_xy_m`、`perturb_yaw_deg` | 27 種擾動 |
| 夾爪位姿指的 frame | `robot.tool_frame` | 必須和上游給位姿時用的 frame 一致 |
| 點雲轉方塊 | `map.*`（體素、分層、方塊填滿率、目標物半徑） | `scene_io.py` |

安裝偏移怎麼量：手臂 joint1 轉軸與 Kachaka 頂面的交點 → 到底盤中心的前後距離 = x、左右 = y（左正）、頂面離地 = z。

## 7. 已知限制

- 手臂安裝偏移 (0.12, 0, 0.124)、Kachaka 尺寸、到位誤差 ±5 cm／±5° 是假設值（config 裡標了 ★）。
- 需確認：SLAM 與 Kachaka 用同一張地圖座標、地板在 z ≈ 0；上游夾爪位姿用的 frame 是 `pinch_center`。
- 點雲轉出的方塊比實物大約 1–2 cm（含安全餘量）；桌下等懸空空間會被當成實心。
- 穩健度用 16 個 IK 種子評分，每次執行會有跳動（同一站位 0.85–0.96），常常走「用最穩健的」退路；可調 `profile.scorer_seeds` 或 `robustness.threshold`。
- cuRobo v2 這個版本的 `collision_checking.validate()` 有 bug，停車檢查改用 cuRobo FK ＋自己算方塊距離（`placement.py:135`）。
- 時間只在 RTX PRO 6000 上量過（solver 建立約 9 s 一次性；找站位到手臂規劃約 1.5 s；GPU 峰值約 0.7 GB），Thor 上需要實測。
