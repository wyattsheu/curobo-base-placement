# curobo-base-placement：用 cuRobo 最佳化找出 Kachaka 該停在哪裡才抓得到

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

在測試場景驗證過（RTX PRO 6000，2026-10-14）：

| 輸入 | 障礙方塊 | 目標物判定 | IK 解 | 選中站位 (map) | 穩健度 |
|---|---|---|---|---|---|
| 點雲＋語意＋目標遮罩 | 12 | `slam_mask` | 43 | (3.16, 0.89, 137°) | 0.63 |
| 點雲＋目標遮罩 | 18 | `slam_mask` | 41 | 同上 | 0.67 |
| 只有點雲（不給目標） | 24 | `radius_fallback` | 41 | 同上 | 0.67 |

三種輸入選到同一個站位。穩健度都沒到 1.0，是走「用最穩健的」退路選出來的（見 2.2）。

---

### 1.5 直接接實驗室的系統（Thor 上的格式）

| 實驗室有的東西 | 怎麼給 | 說明 |
|---|---|---|
| 匯出的 3D 語意地圖資料夾（`robot_ws/data/lab/<site>/`）| `--lab-map <資料夾>` | 讀 `*_sem_color_fine.npz`（points）、`*_instance_labels.npz`、`*_instance_semantic_table_fine.json`、`alignment_*.yaml`；依 alignment（object_query 同一套公式）轉到 Kachaka 地圖座標並把地板移到 z = 0（`lab_map.py`）|
| Kachaka 的 2D 地圖 `occupancy_<site>.yaml/png` | 和 `--lab-map` 一起自動讀，或 `--occupancy <yaml>` | 站位只能落在它的 free 格子上（3D 點雲常常沒有地板點）；Stage1 用的也是這張 |
| object_query 的實例編號 | `"target": {"instance_id": 13}` | 該實例的點就是目標物 |
| 夾爪位姿在 Piper 基座座標（mm_actions 的慣例）| `"grasp_pose_in_piper_base": {...}` 取代 `grasp_pose_in_map` | 用 `base_pose_in_map` 和安裝位置換到地圖座標（同 `piper_point_to_kachaka`）|
| 類別表 `{"segmentation": [...]}` 或 `{"labels": [...]}` | 自動判斷 | `scene_io.load_class_names` |

在 803 室 9/30 地圖上的實測（夾爪位姿為手動標註，見 `real_maps/map803_0930/requests/`，不在本專案內）：瓶子 19 成功、穩健度 1.00；
瓶子 13 成功但穩健度 0.11（夾爪離桌面方塊只剩 3 mm；加了 `min_accept: 0.5` 之後會改回 `not_robust`，尚未重跑）；瓶子 24 在後排被前排瓶子與紙箱圍住，夾爪本身在任何抓法下都會碰到，無解。

## 2. 輸出

`run_demo.py` 印出一行 `@@JSON@@{…}`。上游只要先看 `ok` 和 `reason`。

### 2.1 欄位

| 欄位 | 意義 |
|---|---|
| **`ok`** | true = 找到站位 |
| **`reason`** | `ok`，或沒有結果時卡在哪一步（見 2.3） |
| **`chosen_map`** | **(x, y, yaw) 地圖座標 → 直接給 Kachaka `Navigate_to_pose`**（只有 `ok` 時才有） |
| `chosen_base` | 同一個站位，相對於現在位置（base 座標）的移動量 |
| **`at_goal.grasp_pose_in_piper_base`** | 站在 `chosen_map` 時，上游的夾爪位姿在 Piper 基座座標的值 → 可直接給 mm_actions 的 `move_arm_to_pose` |
| `robustness` | 選中站位的穩健度：假設 Kachaka 停歪 ±5 cm、±5°（27 種），手臂還抓得到的比例 |
| `below_threshold` | true = 沒有站位達到 `threshold`，選中的是退路（見 2.2） |
| `max_robustness` | 評過分的站位中最高的穩健度（`not_robust` 時看差多少） |
| `timing_s`、`memory_mb` | 每一步耗時（秒）與 GPU 記憶體（MB） |
| `n_boxes`、`boxes_base` | 從點雲算出的障礙方塊（base 座標） |
| `target_points`、`target_source` | 排除了幾個目標物的點、用哪種方式（`slam_mask` / `class_near_grasp` / `radius_fallback`）|
| `known_source` | 可停車區域的依據：`occupancy`（Kachaka 2D 地圖）或 `slam_points` |
| `n_success`、`n_cone`、`parked_ok`、`n_checked` | 每一步剩下多少候選：IK 解數、護欄內的解數、每個候選能不能停、評過分的站位數 |
| `q_success`、`uniq`、`rob` | 中間結果（候選站位、評分），除錯與可視化用 |

### 2.2 穩健度門檻：`threshold` 和 `min_accept`（`config.yaml` 的 `robustness`）

1. 由近到遠評分，第一個穩健度 ≥ `threshold`（預設 1.0，27 種全過）的站位就採用。
2. 沒有站位達到 `threshold` 時，採用評過分的站位中最穩健的（同分取最近），但不能低於 `min_accept`（預設 0.5），
   `below_threshold = true`。
3. 都沒有就 `ok = false`、`reason = not_robust`，不會把只有一兩種擾動抓得到的站位當成功回傳。

所以不需要手動把 `threshold` 降到 0.9：退路會自動拿到這個場景最高的分數，`min_accept` 決定最低能接受多少。

### 2.3 `reason`：沒有結果時卡在哪一步

程式每一步都有上限（IK 最多 `wb_rounds_max` 輪、只評有限個候選、退路只試 ≥ `min_accept` 的），不會一直算下去。
`reason` 是**第一個把候選篩光的步驟**：

| reason | 意思 | 上游可以怎麼做 |
|---|---|---|
| `ok` | 找到站位 | 導航到 `chosen_map` |
| `no_ik_solution` | 搜尋範圍內沒有任何 9 軸 IK 解：搆不到，或夾爪位姿本身就碰撞 | 換抓法 |
| `outside_cone` | 有解，但都超出 ±30° 護欄（`robot.bearing_limit_rad`） | 換抓法，或放寬護欄 |
| `cannot_park` | 有解，但底盤停的位置都不在可行駛區域（Kachaka 2D 地圖或 SLAM 看過的地板） | 檢查地圖，或換抓法 |
| `not_robust` | 停得進去，但穩健度都低於 `min_accept`；看 `max_robustness` 差多少 | 冒險降 `min_accept`，或換抓法 |

注意：
- `no_ik_solution` 是取樣沒找到（128 種子 × 最多 4 輪），不是證明無解；解空間很窄時可能漏掉。
  「夾爪本身碰撞」目前也歸在這裡，還沒有分開判斷。
- 底盤的碰撞球在 9 軸 IK 裡就已經檢查過（`robot_model.py:92`），所以地上有障礙物時通常是 `no_ik_solution`；
  `cannot_park` 實際上只會來自可行駛區域的檢查。

每一種 reason 都用場景驗證過，見第 8 節。

### 2.4 到位後：手臂軌跡是另一個步驟

`plan()` 只回答「要移到哪裡」，**不做手臂軌跡規劃**。Kachaka 到位、上游用鏡頭重新確認目標的座標之後，
用**實際到達的位置**和新的夾爪位姿送一個新請求，只規劃手臂：

```bash
python run_demo.py --lab-map <map> --request arrived.json --arm      # 或 --serve 時那一行加 "arm": true
```

程式裡是 `planner.set_scene(新場景)` 再 `planner.plan_arm()`（底盤鎖在現在的位置，6 軸 MotionPlanner 從收起姿態開始）。
輸出 `arm.traj`（N × 6，joint1–6，弧度）、`arm.dt`；規劃不出來時 `reason = arm_plan_failed`，可以檢查 `robot.stow_arm` 或重試。
MotionPlanner 在第一次呼叫時才建立（約 2–3 s）；`profile.build_arm_planner: true` 則在啟動時就建好。

---

## 3. 怎麼跑

### 建構環境（x86_64 + NVIDIA GPU，第一次）

用 [uv](https://docs.astral.sh/uv/) 建一個 Python 3.12 的虛擬環境。以下步驟在 RTX PRO 6000（CUDA 12.8 驅動）上從零跑過：

```bash
uv venv --python 3.12 ~/env_curobo
source ~/env_curobo/bin/activate

# 1. PyTorch（CUDA 12.8 版）
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# 2. cuRobo v2，固定在這個專案測過的 commit
git clone https://github.com/NVlabs/curobo.git
git -C curobo checkout 4ea7736
uv pip install "./curobo[cu12]"

# 3. 這個專案自己的相依
uv pip install -r requirements.txt

# 檢查：應印出 0.8.0 2.11.0+cu128 True
python -c "import curobo, torch; print(curobo.__version__, torch.__version__, torch.cuda.is_available())"
```

- 驅動是 CUDA 13 的話，第 1 步換成 cu13 的 PyTorch，第 2 步用 `"./curobo[cu13]"`（見 cuRobo 的 `docs/getting-started/installation.rst`）。
- Jetson Thor（aarch64）要用 NVIDIA 給 Jetson 的 PyTorch，不能直接照上面裝；cuRobo 部分相同。
- 第一次執行會編譯 warp kernel，比之後慢。

### 有 cuRobo + GPU 的機器（Thor、工作站）

```bash
python run_demo.py                                                   # test_maps/slam.npz + test_maps/request.json
python run_demo.py --npz /maps/xxx.npz --request request.json
# 常駐：solver 只在啟動時建一次（Thor 約 12 s），之後每行一個請求（request.json 路徑，或請求本身的 JSON，
# 可帶 "lab_map" / "npz" / "occupancy" 換地圖，地圖只讀一次），每個請求印一行 @@JSON@@
python run_demo.py --lab-map /maps/auto_20260826 --serve < requests.txt
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
curobo_base_placement/
├─ config.yaml        ← 所有假設值與參數（量到實際數值只改這裡）
├─ run_demo.py        ← 入口：讀點雲與請求 → 規劃 → 印出 JSON
├─ scene_io.py        ← 點雲 + 請求 → 規劃器要的格式（障礙方塊、已知區域、base 座標的夾爪位姿）
├─ robot_model.py     ← 9 軸 cuRobo 模型（Kachaka x, y, ψ + Piper 6 軸）
├─ placement.py       ← 找站位並確認抓得到
├─ bundle.py          ← 打包成一支腳本丟到 server 跑
├─ make_test_maps.py  ← 從 test_scene.yaml 做出測試用的點雲與請求
├─ test_scene.yaml、test_maps/   ← 測試場景（桌上罐子前有擋板、左邊櫃子、後方與右方是牆）
├─ validation/       ← 驗證與效能測試（獨立資料夾，主程式不引用；見第 8 節與 validation/README.md）
└─ assets/            ← robot129.yml（cuRobo 設定、碰撞球）、robot129_description/（URDF、STL）
```

## 5. 資料流（檔案:行號）

```
slam.npz + request.json（map 座標）   或   --lab-map 實驗室匯出資料夾（lab_map.py）
   │ scene_io.py:330 scene_from_slam()
   │   :383 目標物點（遮罩 / instance id → 類別 → 圓柱退路，再往外長 3 cm）   :367 已知區域（或 :317 Kachaka 2D 地圖）   :402 障礙點
   │   :237 objects_to_boxes()  依類別 → 3D 連通物體 → :216 依高度分層 → :171 decompose() 包方塊（不填目標物的位置）
   ▼
PlanningScene（base 座標 = Kachaka 現在的位置）
   │ placement.py BasePlacementPlanner(cfg)   開機時建好 solver；每個請求 set_scene(scene) 只把方塊寫進 tensor
   │ placement.py plan()
   │   1 whole_body()   9 軸 IK：同一個夾爪位姿 → 很多候選站位 (x, y, ψ)
   │   2 _candidates()  ±30° 護欄（robot_model.in_safety_cone）＋ parking_ok() 底盤不撞、在可行駛區域
   │   3 _lm_pass() / _finish()   由近到遠，27 種到位誤差各重解 6 軸 IK（先 LM，需要時才最佳化）；
   │                    第一個 27/27 的就是答案（都沒有時用最穩健的，但不低於 min_accept）
   ▼
chosen_map → Navigate_to_pose；at_goal.grasp_pose_in_piper_base → 到位後的手臂
   │ 到位、重新確認目標後（另一個步驟）：set_scene(新場景) → plan_arm() → arm.traj
```

| 座標系 | 定義 | 轉換 |
|---|---|---|
| map | SLAM／Kachaka 地圖 | 輸入檔案都在這裡 |
| base | Kachaka **現在**的位置：底盤中心投影到地面，x 朝車頭、z 朝上 | `scene_io.py:32 map_to_base_xy`、`:49 quat_map_to_base`、`:43 base_to_map_pose`（輸出給導航）|

## 6. 量到實際數值要改哪裡（`config.yaml`）

| 量到的東西 | 欄位 | 進到程式的哪裡 |
|---|---|---|
| 手臂裝在 Kachaka 的位置 | `robot.arm_mount_xyz`、`arm_mount_rpy_deg` | `robot_model.py:91`（運動鏈）、`:133`（護欄）|
| Kachaka 實際尺寸 | `robot.kachaka.size_m` | 碰撞球、停車檢查 |
| 相機桿等凸出物 | `robot.kachaka.extra_spheres` | 碰撞球 |
| 開車時手臂收起的姿態 | `robot.stow_arm`（目前用實驗室 mm_actions 的 home 姿態：全零）| 停車檢查、手臂規劃起點 |
| Kachaka 到位誤差 | `robustness.perturb_xy_m`、`perturb_yaw_deg` | 27 種擾動 |
| 夾爪位姿指的 frame | `robot.tool_frame` | 必須和上游給位姿時用的 frame 一致 |
| 點雲轉方塊 | `map.*`（體素、分層、方塊填滿率、目標物半徑） | `scene_io.py` |

安裝偏移怎麼量：手臂 joint1 轉軸與 Kachaka 頂面的交點 → 到底盤中心的前後距離 = x、左右 = y（左正）、頂面離地 = z。

## 7. 已知限制

- 手臂安裝位置 (0.08, 0, 0.525)、Kachaka 車身（0.387 × 0.240 × 0.095 m，中心在轉軸前 4.35 cm）、工具點（gripper_base 前 0.12 m）
  取自實驗室的 `kachaka_piper.urdf.xacro` 與 mm_actions。底盤上方撐起手臂的支架（和車身同樣大小、到 0.525 m）不建模：
  底盤只問「能不能停」（底盤的碰撞球 + Kachaka 2D 地圖），碰撞只在規劃手臂時檢查。右側相機桿尚未加入；到位誤差 ±5 cm／±5° 是假設值。
- 收起姿態用 mm_actions 夾到東西後的姿態；空手的 home（全零）完全折疊，robot129 的碰撞球在那裡重疊，cuRobo 會拒絕當起點。
- 需確認：SLAM 與 Kachaka 用同一張地圖座標、地板在 z ≈ 0；上游夾爪位姿用的 frame 是 `pinch_center`。
- 點雲轉出的方塊比實物大約 1–2 cm（含安全餘量）；桌下等懸空空間會被當成實心。
- 穩健度用 8 個 IK 種子評分（見第 9 節），每次執行仍可能有小幅跳動；可調 `profile.scorer_seeds`、`robustness.threshold` 或 `robustness.min_accept`。
- 停車檢查只用底盤的 10 顆球，直接用 numpy 算到方塊的距離（`placement.py:141`）；cuRobo v2 這個版本的 `collision_checking.validate()` 有 bug，沒有使用。
- 時間見第 9 節（Thor + 實驗室真實地圖）。

## 8. 驗證場景

把測試場景（`test_maps/`）改掉一個條件，確認每一種 `reason` 都在該出現的地方出現：

```bash
python validation/run_scenarios.py                    # 全部場景，每個印 PASS / FAIL
python validation/run_scenarios.py no_drivable_area   # 只跑指定的
```

`validation/` 是獨立的測試資料夾：主程式不引用它，改動只在記憶體裡做（改 config 的值、移動夾爪位姿、把可行駛區域挖掉），
不改 `config.yaml` 或 `test_maps/`。不用的話可以整個忽略。各程式的說明見 `validation/README.md`。

結果（RTX PRO 6000，2026-10-14，9/9 通過）：

| 場景 | 改了什麼 | reason | IK 解 | 護欄內 | 可停 | 穩健度（選中 / 最高） |
|---|---|---|---|---|---|---|
| `baseline` | 不改 | `ok`（退路） | 43 | 33 | 30 | 0.67 / 0.67 |
| `loose_threshold` | `threshold` 降到 0.5 | `ok`（直接達標） | 43 | 33 | 30 | 0.59 / 0.59 |
| `unreachable` | 夾爪位姿抬到離地 2.0 m | `no_ik_solution` | 0 | 0 | 0 | – |
| `gripper_in_cabinet` | 夾爪位姿放進櫃子裡（夾爪本身一定碰撞） | `no_ik_solution` | 0 | 0 | 0 | – |
| `cone_closed` | 護欄收到 ±0°（強制觸發這個分支） | `outside_cone` | 181 | 0 | 0 | – |
| `no_drivable_area` | 目標 1.2 m 內都標成不可行駛 | `cannot_park` | 181 | 123 | 0 | – |
| `sloppy_navigation` | 到位誤差假設成 ±15 cm、±15° | `not_robust` | 43 | 33 | 30 | – / 0.26 |
| `arm_plan` | 不改；選好站位後另外呼叫 `plan_arm()` | `ok` | 43 | 33 | 30 | 0.67 / 0.67 |
| `bad_stow` | 收起姿態設成全零（碰撞球自我重疊），另外呼叫 `plan_arm()` | `arm_plan_failed` | 43 | 33 | 30 | 0.67 / 0.67 |

- `loose_threshold` 評了 4 個站位就停（由近到遠，第一組就有達標的），其他場景評完全部 30 個。
- IK 解 181 的兩個場景：可停的候選一直不到 `min_candidates`（15 個），所以 IK 跑滿 `wb_rounds_max`（4 輪）才停。
- `bad_stow` 會印出 cuRobo 的 `Start or End state in collision`，是預期的。
- 穩健度每次執行會跳動，數字可能和上表差一點，但 reason 應該一樣。


## 9. 在 Thor 上用真實地圖：加速與驗證

真實 SLAM 地圖會轉成 350–950 個方塊（測試場景只有 12 個）。原本在 Thor 上，每個請求光建 solver 就要 56 s，
評分每個站位要 4–14 s。改了三件事，選出的站位不變或更好：

1. **只放手臂碰得到的方塊**（`placement.load_envs`、`robot_model.arm_reach`）。以肩關節為中心，任何手臂姿態下
   碰撞球都在 0.887 m 內（三角不等式，可證明的上界）；更遠的方塊離任何球都超過 cuRobo 的啟動距離 0.01 m，
   對碰撞成本的貢獻精確為 0。`validation/check_near_exact.py` 用 cuRobo 自己的 FK 驗證：40 萬個隨機手臂姿態
   最遠 0.770 m；513 個真實擾動站位中，被拿掉的方塊離碰撞球最近 0.169 m。每個站位從 942 個方塊降到中位數 135 個。
2. **評分改成「LM 種子 ∪ 最佳化結果」**（`placement.solve_envs`）。cuRobo 的 IK 先用 Levenberg-Marquardt 從隨機起點解位姿，
   再用 MPPI + L-BFGS 處理碰撞。實測第二段會把許多「本來就有效」的 LM 解變成失敗，而且種子越多、在 batch 裡越後面越嚴重：
   同一個站位複製 4 份放進同一批，得到 18/12/1/0 個成功（滿分 27）。兩者都經過同一套 metrics
   （碰撞、關節限制、位姿誤差）檢查，所以任一邊有有效解就算到得了；所有 env 都已經有解時就不跑最佳化。
   改完後，分數和每次呼叫放幾個站位無關，重複執行的跳動也幾乎消失。
3. **常駐模式**（`run_demo.py --serve`、`BasePlacementPlanner.set_scene`）：solver 只建一次，容量固定
   （`profile.max_scene_boxes` / `max_near_boxes`），每個請求只把方塊寫進 tensor。容量固定之後，手臂規劃也打開了
   CUDA graph（Thor 上 1.0–1.4 s → 0.24–0.35 s，選到的站位相同）。
4. **安全錐外的擾動不觸發最佳化**：它們不管 IK 結果如何都算失敗，所以 LM 種子沒解出來時，也不會因為它們去跑最佳化。

種子數（`validation/bench_seeds.py`，基準是 32 種子、一次一個站位）：在 ec129 與 auto_20260826 上，16／8／4 種子選出的站位都和基準相同，
regret 都是 0，pass 判定一致率 1.00，所以預設改成 8 個。

| Thor，實驗室真實地圖 | 修改前 | 修改後 |
|---|---|---|
| 建 solver | 每個請求 56 s | 常駐只在啟動時一次，11.9 s |
| 評分（每個站位） | 8.7–13.6 s（16 種子） | 0.36–0.58 s（8 種子） |
| 手臂軌跡規劃 | — | 0.24–0.35 s |
| 一個請求（常駐，auto_20260826 / ec129，12 次） | 數分鐘 | 2.6–17.4 s，通常 3–6 s；成功 12/12 |

常駐模式下，剩下的時間主要是評分（每個站位約 0.5 s；最壞的情況要評 27 個站位），以及從點雲建場景
（CPU 約 1.2 s；第一次讀地圖約 5–6 s）。

```bash
python validation/bench_near.py <地圖根目錄> <site>                  # 全部方塊 vs 只放附近：分數差、時間（各重複 3 次）
python validation/check_near_exact.py <地圖根目錄> <site> <bench_near 的輸出>   # 被拿掉的方塊確實碰不到
python validation/bench_seeds.py <地圖根目錄> <site> <bench_near 的輸出>        # 種子數／warm start 對 32 種子基準
```

### Profiling：時間花在哪（`validation/profile_plan.py`）

Thor 上跑了三張地圖（auto_20260826、ec129、map803），共 60 個請求，其中 15 個找到站位。數字是這 15 個請求的中位數：
建場景 1.22 s、整體 IK 0.04 s、**評分 3.17 s**、手臂規劃 0.28 s，合計約 4.7 s。

- 評分時間的 99% 是 IK 第二段的最佳化（MPPI + L-BFGS），每次呼叫 1.87 s；LM 解加上檢查只要 0.01 s。
- 43 次評分呼叫全部觸發最佳化。觸發時，LM 沒解出來的擾動中位數是 38/101。
- 最佳化在 LM 沒解出來的 2120 個擾動中只救回 62 個（2.9%）。159 個站位裡只有 1 個靠它才達到 27/27，
  18 個請求裡只有 1 個因此選到不同的站位。

所以最佳化不能整個拿掉（會改變結果），但可以只在可能改變答案的地方跑：只對 LM 沒解出來的擾動跑；
或先用 LM 評完所有站位，只對比「第一個 LM 就 27/27 的站位」更近的站位跑最佳化。

### 最佳化只用在可能改變答案的地方（`profile.lazy_optimiser`，預設開啟）

依照上面的 profiling，評分改成兩步（`placement._lm_pass`、`placement._finish`），選站規則不變（由近到遠，第一個 27/27）：

1. **先用 LM 評完所有候選站位**：只跑 IK 第一段加上檢查，每次呼叫約 0.01 s。
2. **需要某個站位的最終分數時**，只把它 LM 沒解出來、而且在安全錐內的擾動，連同後面站位的，裝滿另一個小批次的求解器
   （`profile.optimiser_batch`，27 格）去跑最佳化。LM 已經 27/27 的站位不需要最佳化；比選中站位更遠的站位也不會被算。

`validation/bench_lazy.py`，Thor，三張地圖，60 個請求：

| 設定 | 和舊做法選到相同結果 | 評分時間總和 | 評分中位數 | 規劃中位數（找到站位的請求） |
|---|---|---|---|---|
| 舊做法 | — | 81.5 s | 3.21 s | 3.60 s |
| 舊做法再跑一次 | 55/60（舊做法本身的變化） | 92.4 s | 4.41 s | 5.34 s |
| **新做法，批次 27（預設）** | **59/60** | **38.1 s** | **1.30 s** | **1.61 s** |
| 新做法，批次 54 | 55/60 | 29.7 s | 1.18 s | 1.52 s |

找到站位的請求兩邊都是 15 個。新做法和舊做法的差異比舊做法自己重跑的差異還小；最慢的請求從 13.4 s 降到 6.6 s。
最佳化的結果會受到「哪些問題放在同一批」影響；把不同站位的擾動放進原本的大批次時，60 個請求裡有 4 個結果不同，
所以最佳化改用自己的小批次。

### 建場景用多核心（`map.box_workers`，預設 8）

點雲切方塊時，每個物體各自獨立計算（`scene_io._object_boxes`），所以改成分給多個 CPU 行程同時算。
行程池在第一次用到時建立，之後一直沿用；結果依原本的順序組回，每個方塊的名稱、位置、大小、角度都和不平行時**完全相同**
（6 個真實請求逐一比對，2／4／8 個 worker 都相同）。

Thor，三張地圖，60 個請求（`validation/profile_plan.py`，最佳化只用在可能改變答案的地方 + 8 個 worker）：

| | 之前 | 現在 |
|---|---|---|
| 整個請求（找到站位的 15 個，中位數） | 4.74 s | **1.95 s** |
| 最慢的請求 | 14.8 s | 5.7 s |
| 建場景 | 1.22 s | 0.26 s |
| 評分 | 3.17 s | 1.28 s |
| 手臂規劃 | 0.28 s | 0.28 s |

之後手臂軌跡規劃移出 `plan()`（見 2.4），同樣 60 個請求：找到站位的 15 個中位數 **1.66 s**，10/15 在 2 s 內，最慢 5.8 s。

### 只把手臂碰得到的地圖切成方塊？（`map.crop_to_window`，預設關閉）

試過在切方塊之前，就丟掉「搜尋視窗內任何站位的手臂都碰不到」的點（`scene_io.window_reach_mask`），
同時把搜尋視窗縮小，對應「機器人已經在目標旁邊、只小幅移動」的情境。`validation/bench_roi.py`：auto_20260826 + ec129，
每張地圖 10 個夾爪位姿 × 2 種起點距離，RTX PRO 6000：

| 設定 | 方塊數（中位數） | 建場景 | 有解的請求（起點 0.6–0.9 m / 0.9–1.3 m） | 選到的站位和現在相同 |
|---|---|---|---|---|
| 視窗 ±1.5 m，不裁切（現在） | 666–942 | 0.77–0.86 s | 6 / 6 | 重跑 12/12 相同 |
| ±1.5 m + 裁切 | 650–939 | 0.77–0.89 s | 6 / 6 | 10/12 |
| ±0.8 m + 裁切 | 494–786 | 0.62–0.83 s | 6（少 1）/ 6 | 1/11 |
| ±0.5 m + 裁切 | 322–606 | 0.45–0.61 s | 7 / 1（少 5） | 1/7 |

- 現在的 `roi_radius_m` 1.8 m 本來就差不多等於「手臂碰得到的範圍」，所以視窗不變時方塊幾乎沒少，速度也沒變。
- 視窗縮到 ±0.5 m，建場景只省約 0.3 s；起點稍遠時就找不到站位，選到的站位穩健度也較低。
- 就算視窗不變，物體在裁切邊界被切斷後，方塊會重新擬合，結果會跟著變；不裁切時重跑結果完全相同。

Thor 上（auto_20260826、ec129、map803，共 240 個請求）結論相同，而且更明顯：建場景從 1.22 s 最多只降到 0.73–0.89 s，
但整個請求的中位數從 4.6–4.7 s 變成 4.9–20.3 s。視窗變小之後好的候選站位變少，要評分的站位反而變多。

結論：速度收益小，又會改變結果，所以預設關閉。

## 10. 修改前的基準（commit `11d341b`，2026-10-14）

第 9 節的修改**之前**的量測，留作對照：結論已被第 9 節取代。在 Jetson Thor（lab173，實驗室 `mm-gpu` image 內的
cuRobo 0.8.0.post1）上量，測試程式在 `validation/`。

### 10.1 測試場景，20 個夾爪位姿（`bench_multi_grasp.py`）

同一個罐子，接近方向每 36° 一個、水平與往下 30° 兩種，共 20 個；只有 4 個有可停的站位。

| 做法 | 20 個總時間 | 選中 |
|---|---|---|
| 每個夾爪位姿各跑一次 `plan()`（含手臂規劃），再挑最好的 | 14.3 s | #13，穩健度 0.96 |
| 全部評分完再挑，只對選中的做手臂規劃 | 6.3 s | 同上 |
| ＋第一輪 IK 就 0 解的夾爪位姿直接放棄（不重試 4 輪） | 3.6 s | 同上 |

- 時間主要花在穩健度評分；whole-body IK（128 種子）每個夾爪位姿只要約 7 ms。whole-body 種子降到 32 會漏掉好站位，不要降。
- Thor 的 GPU 已經算滿：每批放更多站位（4 → 32）反而更慢、更吃記憶體，所以把多個夾爪位姿合併成同一批沒有好處。
- 同一個測試場景，Thor 比 RTX PRO 6000 慢約 4.5 倍（每個站位 136 ms vs 30 ms）。

### 10.2 真實地圖：評分種子能不能減少（`bench_lab_maps.py`）

實驗室 5 張地圖、每張 2 個桌上物體、每個物體 20 種抓法。同一批候選站位分別用 16（現在的設定）、8、4 個評分種子評分，
以 32 種子為基準。品質損失 = 基準的最佳分數 − 基準替該設定選中的站位打的分數。

12 個場景中 6 個有效；另外 6 個（ec129_0911_bridge、ec129_manual_0813b、new_P_1908）20 種抓法都沒有候選站位，
這三張地圖沒有 `instance_labels.npz`，地板與物體位置是估的，應是測資準備的問題，不列入比較。

| 評分種子 | 6 個場景總評分時間 | 選錯站位的場景 | 最大品質損失 | 平均誤差 | 可接受判斷一致率（最差／平均） |
|---|---|---|---|---|---|
| **16（現在）** | 970 s | 0 | 0 | 0.018 | 86% ／ 96% |
| 8 | 482 s | 1 | 0.30 | 0.025 | 79% ／ 94% |
| 4 | 130 s | 1 | 0.22 | 0.041 | 78% ／ 90% |

- 簡單場景（很多站位穩健度 1.0）種子多少都選對；擁擠的場景（ec129_0813b 瓶子 21，最好的站位 0.889）8 種子把最好的站位低估，
  改選一個只有 0.593 的，分數系統性偏低。原因後來查到是 cuRobo 的最佳化步驟把有效的 LM 解變成失敗；
  第 9 節改成「LM 種子 ∪ 最佳化結果」之後，8 種子與 32 種子選出相同站位，所以預設改為 8。
- 真實地圖的障礙方塊遠比測試場景多（28–945 個 vs 12 個），評分時間大約和方塊數成正比：945 個方塊時 16 種子每個站位約 13 s。

### 10.3 手臂碰不到的方塊（`reach_cull.py`，只用 CPU）

以夾爪為中心的保守上界（手臂 1.07 m、底盤水平 1.12 m 且低於 0.105 m）只做計數：12 個場景平均可拿掉 61%（3899 → 1536 個）。
第 9 節以肩關節為中心的做法（0.887 m，每個站位各自篩）更緊，也已用 cuRobo 的 FK 驗證，實際採用的是那個。
