# validation/：驗證與效能測試

這個資料夾只放測試用的程式。主程式（`run_demo.py`、`placement.py`、`scene_io.py`…）完全不引用它，
所有場景的改動都只在記憶體裡做，不改 `config.yaml` 或 `test_maps/`。不用的話可以整個忽略。

都需要 cuRobo + GPU（環境見上層 README 第 3 節），從專案根目錄執行：

| 程式 | 做什麼 | 結果寫在 |
|---|---|---|
| `run_scenarios.py` | 測試場景改掉一個條件，確認每一種 `reason` 都在該出現的地方出現（8 個場景，印 PASS / FAIL） | 上層 README 第 8 節 |
| `bench_multi_grasp.py` | 上游給多個夾爪位姿時的耗時：`sweep` 評分種子數／批次大小、`warm` warm start、`e2e` 20 個夾爪位姿從頭到尾 | 上層 README 第 9 節（待補） |
| `bench_lab_maps.py` | 在實驗室的真實 SLAM 地圖上，少量種子選出的站位是否和 32 種子的基準一樣好 | 上層 README 第 9 節（待補） |

```bash
python validation/run_scenarios.py                        # 全部場景
python validation/run_scenarios.py no_drivable_area       # 只跑指定的
python validation/bench_multi_grasp.py e2e seeds=8 group=4 k=8 stop0=1
python validation/bench_lab_maps.py <實驗室地圖根目錄> [site ...]   # 例：robot_ws/data/lab
```

在 Jetson Thor 上可以用實驗室的 `mm-gpu` image（已有 cuRobo）跑，不用另外裝：

```bash
docker run --rm --runtime nvidia --gpus all -v $PWD:/work -w /work \
  -v <實驗室地圖根目錄>:/maps:ro --entrypoint python3 mm-gpu:latest validation/bench_lab_maps.py /maps
```
