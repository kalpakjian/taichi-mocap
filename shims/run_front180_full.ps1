# Full-clip FRONT view (90deg + 180deg = 270deg). Renders into a separate folder so the
# back-view (camL90) deliverables are untouched.
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\princ\Downloads\taichi_enh'
$src = "$root\taichi_full_30fps.mp4"
$status = "$root\front180_status.txt"
$b = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$env:PYTHONPATH = 'C:\Users\princ\shims'

function Step($msg) { "[$(Get-Date -Format 'HH:mm:ss')] $msg" | Tee-Object -FilePath $status -Append }

Set-Content -Path $status -Value "start $(Get-Date -Format 'HH:mm:ss')"
Step 'STEP 1/2 blender render front180 (270 deg, static)'
New-Item -ItemType Directory -Force -Path "$root\preview_gvhmr_full\front180\frames" | Out-Null
$env:BVH_PATH = "$root\mocap\gvhmr_full.bvh"
$env:OUT_DIR = "$root\preview_gvhmr_full\front180"
$env:NPZ_PATH = "$root\mocap\gvhmr_full.npz"
$env:PREVIEW_END = '0'
$env:RES = '960x540'
$env:FPS = '30'
$env:CAM_ANGLE = '270'
$env:CAM_SIDE = 'left'
& $b -b --factory-startup -P 'C:\Users\princ\scripts\bvh_preview_blender.py' *>> "$root\front180_blender.log"
if ($LASTEXITCODE -ne 0) { Step "FAILED blender exit=$LASTEXITCODE"; exit 1 }

Step 'STEP 2/2 encode mp4s (silent)'
& ffmpeg -v error -y -framerate 30 -i "$root\preview_gvhmr_full\front180\frames\f_%04d.png" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -an "$root\preview_gvhmr_full\stick_full_front180.mp4"
& ffmpeg -v error -y -i $src -i "$root\preview_gvhmr_full\stick_full_front180.mp4" -filter_complex "[0:v]scale=960:540,setsar=1[a];[1:v]scale=960:540,setsar=1[b];[a][b]hstack" -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p "$root\preview_gvhmr_full\gvhmr_full_side_by_side_front180.mp4"

Step 'DONE'
