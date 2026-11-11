# taichi-mocap

把「單鏡頭、低解析度」的太極拳影片轉成 3D 骨架動畫 —— 兩條路線、全部在本機跑完的實測紀錄。

Single-camera motion capture for Tai Chi footage, with two routes: a licence-free MediaPipe
pipeline and a higher-accuracy GVHMR + SMPL-X pipeline. Everything runs locally on one RTX 4090.

## 結果

| | Route A：MediaPipe（免授權） | Route B：GVHMR + SMPL-X |
|---|---|---|
| 關節數 | 17 | 22（SMPL 標準） |
| 骨長一致性 | 每幀浮動，差數十公分 | **0.00 cm**（剛體） |
| 骨架還原誤差 | **37 cm** | **0.03 cm** |
| 軀幹扭轉（twist） | 無 | 有 |
| 60 秒素材處理時間 | 約 40 秒（CPU） | 前置 ~100 秒 + 推論 1.1 秒（GPU） |
| 10 分 33 秒全片 | — | 前置 864.6 秒 + 推論 79.7 秒 |

素材本身：640×360、419 kbps、15 fps、9,497 幀。極限不是模型，是素材（單眼、寬鬆衣服、腳被字幕遮住）。

## 流程

```
原始 FLV (640x360, 419kbps)
  └─ upscale_video.py         Real-ESRGAN x2 fp16 → 1280x720 / 2.25 Mbps   [GPU]
       ├─ Route A ─ pose_extract.py      MediaPipe 33 landmarks → 17 關節 BVH + npz  [CPU]
       │              verify_bvh.py      FK 回歸檢查（方向 cos / 位置誤差）
       │              reproject_check.py 把 3D 投回原片，量投影誤差
       └─ Route B ─ GVHMR demo           YOLO + ViTPose + HMR2 + denoiser      [GPU]
                      gvhmr_export.py    SMPL-X 參數 → 22 關節 BVH（保留 twist）
                      verify_bvh.py      同上
  └─ bvh_preview_blender.py    Blender headless（Workbench）算骨架預覽，可設方位角  [GPU]
```

## 快速開始

### Route A（不需要註冊任何授權）

```powershell
python -m pip install mediapipe opencv-python numpy

# 影片 → 骨架（60 秒片段）
python scripts/pose_extract.py INPUT.mp4 outdir --start 100 --seconds 60 --smooth 5 --tag take60

# 回歸檢查：BVH 必須能還原出同一組關節位置
python scripts/verify_bvh.py outdir\take60.bvh outdir\take60.npz

# 把 3D 骨架投影回原片（肉眼驗收）
python scripts/reproject_check.py INPUT.mp4 outdir\take60.npz outdir\reproject.mp4
```

輸出：`take60.bvh`（17 骨、cm、Y-up、腳貼地）、`take60.npz`（世界座標 + 2D landmarks + 可見度）。

### Route B（GVHMR + SMPL-X，需要 SMPL-X 授權）

1. 依 `shims/README.md` 在 Windows 上裝好 GVHMR（含純 torch 的 pytorch3d 替代品）。
2. 取得 `SMPLX_NEUTRAL.npz`（非商用授權），用 `shims/install_body_models.ps1` 放到正確位置。
3. 跑 demo 與匯出：

```powershell
cd C:\path\to\GVHMR
$env:PYTHONPATH = 'C:\path\to\taichi-mocap\shims'
python tools/demo/demo.py --video CLIP30FPS.mp4 -s --no_render

python ..\taichi-mocap\scripts\gvhmr_export.py `
  outputs\demo\CLIP30FPS\hmr4d_results.pt outdir --tag gvhmr60 --fix-root
```

- `-s` = 固定機位（跳過視覺里程計）
- `--no_render` = 跳過網格預覽（shim 沒有渲染器，動作輸出不受影響）
- `--fix-root` = 扣掉水平漂移（靜止機位＋單眼深度會讓人物看起來在走動）

### 骨架預覽

```powershell
$env:BVH_PATH='outdir\gvhmr60.bvh'; $env:NPZ_PATH='outdir\gvhmr60.npz'
$env:OUT_DIR='preview'; $env:FPS='30'; $env:CAM_ANGLE='90'; $env:CAM_SIDE='left'
blender -b --factory-startup -P scripts\bvh_preview_blender.py
```

`CAM_ANGLE` 以「第一幀的正面」為基準旋轉，`CAM_SIDE=left|right` 換另一側；`CAM_ANGLE=270` 等於 90° 再轉 180°（正面）。

## 檔案

| 檔案 | 用途 |
|---|---|
| `scripts/upscale_video.py` | 串流式 GPU 超分（解碼→去噪→ESRGAN→編碼，不落地中繼檔） |
| `scripts/pose_extract.py` | MediaPipe 33 landmarks → 17 關節 BVH / npz（自帶骨架疊加影片） |
| `scripts/mocap_math.py` | 共用的旋轉數學（align / Euler ZXY），兩條路線共用 |
| `scripts/verify_bvh.py` | FK 回歸檢查：BVH 通道還原成關節位置，比對方向 cos 與位置誤差 |
| `scripts/reproject_check.py` | 單應 PnP 解相機後把 3D 投回原片，輸出可目視的驗收影片 |
| `scripts/check_pose_pnp.py` | 只跑 PnP，比較「鏡像 vs 修正」兩種座標系的重投影誤差 |
| `scripts/facing_test.py` | 用「鼻子是否比耳朵靠近相機」判定動作是否被鏡像 |
| `scripts/gvhmr_export.py` | SMPL-X 參數 → 22 關節 BVH（直接用 axis-angle，保留 twist） |
| `scripts/bvh_preview_blender.py` | Blender 無頭渲染骨架，自動取景、方位角可調 |
| `shims/pytorch3d/` | 純 torch 的 pytorch3d 替代品（Windows 沒官方 wheel 的解方） |
| `shims/*.ps1` | Windows 安裝腳本：GVHMR 環境、SMPL-X 檔放置、全片流程 |

## 環境（實測版本）

| 元件 | 版本 |
|---|---|
| GPU | RTX 4090 24 GB（驅動 621.14） |
| Python（MediaPipe 路線） | 3.12 + mediapipe 1.2.2 + opencv 5.0 |
| Python（GVHMR 路線） | 3.12 + torch 2.3.0+cu121 + smplx 0.1.28 + pycolmap 4.2.1 |
| Blender | 5.2.1 LTS（Workbench 引擎，headless） |
| ffmpeg | n8.1（libx264；本專案沒用 NVENC） |

## 資源實測（扣掉桌面基準）

| 階段 | 峰值 VRAM | 用 GPU？ |
|---|---|---|
| Real-ESRGAN 超分 | +0.38 GB | 是（fp16，約 11.6 幀/秒 @640×360→1280×720） |
| MediaPipe 姿態 | 0 | 否（XNNPACK CPU） |
| GVHMR 全流程 | +3.9 GB（峰值 5.35 GB） | 是（瞬時可到 76%） |
| Blender Workbench 渲染 | +0.03 GB | 是（平均 24%，約 24 幀/秒 @960×540） |
| ffmpeg 編碼 | 0 | 否 |

18,994 幀渲染約 22 分鐘；暫存 PNG 約 9.8 GB（跑完請自行刪除）。

## 踩過的坑（留給下一個自己）

1. **手性**：把 MediaPipe 的 y-down 只翻 y 軸會讓座標系變左手系（det = −1），整個動作變鏡像。要翻 y 與 z。判定方法見 `facing_test.py`（鼻子在前比例 0.18 → 0.83）。
2. **BVH 通道順序**：通道值必須依「階層宣告順序」排列，不是依你的關節陣列順序，否則旋轉會貼到別的骨頭。
3. **Euler 反解**：ZXY 的展開式要先確認 round-trip 誤差 < 1e-6，錯了 FK 會整體偏移但畫面不一定看得出來。
4. **匯出方式決定精度**：從關節位置反推旋轉會掉 twist（誤差 12.51 cm）；直接用 SMPL-X 的 axis-angle 可到 0.03 cm。
5. **單眼漂移**：靜止機位＋單眼估深度會讓世界座標水平漂移（實測 3.66 m），用 `--fix-root` 扣掉。
6. **相機方位**：用 `ankle→toe` 當「正面」在這支素材是反的；先出 10 秒 sample 再跑全片。

## 授權與第三方

- 本 repo 的程式碼：MIT（見 `LICENSE`）。
- **SMPL-X 人體模型非商用**，需自行到 smpl-x.is.tue.mpg.de 註冊取得，本 repo 不包含、也不轉散佈。
- `shims/pytorch3d/` 是依 PyTorch3D 公開公式重寫的純 torch 實作（PyTorch3D 本身為 BSD 授權），只涵蓋本流程實際用到的函式。
- GVHMR、ViTPose、HMR2、YOLO 的程式與權重各自有授權，請依原專案條款使用。
- 測試素材（優酷上的太極拳示範影片）**不隨本 repo 提供**，請自備有權使用的影片。

## 相關文章

- [一支 640×360 的太極拳影片，能變成可用的 3D 動作嗎？MediaPipe 與 GVHMR 兩條路線的實測](https://kalpakjian.github.io/2026/11/11/taichi-mocap-single-camera/)
