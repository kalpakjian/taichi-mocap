# Full 74-form pipeline: GVHMR -> BVH -> Blender (90 deg) -> mp4s
# NOTE: all paths ASCII-only; PowerShell 5.1 reads this file as ANSI, so non-ASCII paths break.
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\princ\Downloads\taichi_enh'
$src = "$root\taichi_full_30fps.mp4"
$status = "$root\full_status.txt"
$py = 'C:\Users\princ\venvs\gvhmr312\Scripts\python.exe'
$b = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$env:PYTHONPATH = 'C:\Users\princ\shims'
$env:PYTHONUTF8 = '1'

function Step($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Tee-Object -FilePath $status -Append }

Set-Content -Path $status -Value "start $(Get-Date -Format 'HH:mm:ss')"
cd 'C:\Users\princ\GVHMR'
if (-not (Test-Path $src)) { Step "FAILED missing 30fps source $src"; exit 1 }
Step "source $([math]::Round((Get-Item $src).Length/1MB,1)) MB"

Step 'STEP 2/4 GVHMR demo (full clip, static cam, no render)'
& $py tools/demo/demo.py --video $src -s --no_render *>> "$root\full_gvhmr.log"
if ($LASTEXITCODE -ne 0) { Step "FAILED gvhmr exit=$LASTEXITCODE"; exit 1 }

Step 'STEP 3/4 export BVH'
& $py 'C:\Users\princ\scripts\gvhmr_export.py' 'C:\Users\princ\GVHMR\outputs\demo\taichi_full_30fps\hmr4d_results.pt' "$root\mocap" --tag gvhmr_full --fix-root *>> "$root\full_gvhmr.log"
if ($LASTEXITCODE -ne 0) { Step "FAILED export exit=$LASTEXITCODE"; exit 1 }

Step 'STEP 4/4 blender render (90 deg, static)'
New-Item -ItemType Directory -Force -Path "$root\preview_gvhmr_full\frames" | Out-Null
$env:BVH_PATH = "$root\mocap\gvhmr_full.bvh"
$env:OUT_DIR = "$root\preview_gvhmr_full"
$env:NPZ_PATH = "$root\mocap\gvhmr_full.npz"
$env:PREVIEW_END = '0'
$env:RES = '960x540'
$env:FPS = '30'
$env:CAM_ANGLE = '90'
$env:CAM_SIDE = 'left'
& $b -b --factory-startup -P 'C:\Users\princ\scripts\bvh_preview_blender.py' *>> "$root\full_blender.log"
if ($LASTEXITCODE -ne 0) { Step "FAILED blender exit=$LASTEXITCODE"; exit 1 }

Step 'STEP 5/5 encode mp4s (silent)'
& ffmpeg -v error -y -framerate 30 -i "$root\preview_gvhmr_full\frames\f_%04d.png" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -an "$root\preview_gvhmr_full\stick_full_camL90.mp4"
& ffmpeg -v error -y -i $src -i "$root\preview_gvhmr_full\stick_full_camL90.mp4" -filter_complex "[0:v]scale=960:540,setsar=1[a];[1:v]scale=960:540,setsar=1[b];[a][b]hstack" -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p "$root\preview_gvhmr_full\gvhmr_full_side_by_side_camL90.mp4"

Step 'DONE'
