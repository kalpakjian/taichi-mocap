# Full-clip pipeline: 30fps source -> GVHMR -> BVH -> Blender render -> silent mp4s.
# Needs <Work>/source_30fps.mp4 (make it with: ffmpeg -i IN.mp4 -vf fps=30 -c:v libx264 -crf 20 -an source_30fps.mp4).
param(
    [string]$Work      = $(if ($env:MOCAP_WORK) { $env:MOCAP_WORK } else { Join-Path $HOME 'mocap-work' }),
    [string]$GvhmrRepo = $(if ($env:GVHMR_HOME) { $env:GVHMR_HOME } else { Join-Path $HOME 'GVHMR' }),
    [string]$Venv      = $(if ($env:GVHMR_VENV) { $env:GVHMR_VENV } else { Join-Path $HOME 'venvs\gvhmr312' }),
    [int]$CamAngle     = 90,
    [string]$CamSide   = 'left'
)
$ErrorActionPreference = 'Continue'
$py = Join-Path $Venv 'Scripts\python.exe'
$scripts = Join-Path $PSScriptRoot '..\scripts'
$src = Join-Path $Work 'source_30fps.mp4'
$tag = 'gvhmr_full'
$status = Join-Path $Work 'full_status.txt'
$blender = if ($env:BLENDER_EXE) { $env:BLENDER_EXE } else {
    $c = Get-ChildItem (Join-Path $env:ProgramFiles 'Blender Foundation') -Filter blender.exe -Recurse -ErrorAction SilentlyContinue |
         Sort-Object FullName -Descending | Select-Object -First 1
    if ($c) { $c.FullName } else { 'blender' }
}
$env:PYTHONPATH = $PSScriptRoot
$env:PYTHONUTF8 = '1'

function Step($m) { "[$(Get-Date -Format 'HH:mm:ss')] $m" | Tee-Object -FilePath $status -Append }
New-Item -ItemType Directory -Force -Path $Work | Out-Null
Set-Content -Path $status -Value "start $(Get-Date -Format 'HH:mm:ss')"
if (-not (Test-Path $src)) { Step "FAILED missing 30fps source: $src"; exit 1 }
Step "source $([math]::Round((Get-Item $src).Length/1MB,1)) MB  work=$Work"

cd $GvhmrRepo
Step 'STEP 1/4 GVHMR demo (static cam, no render)'
& $py tools/demo/demo.py --video $src -s --no_render *>> (Join-Path $Work 'gvhmr.log')
if ($LASTEXITCODE -ne 0) { Step "FAILED gvhmr exit=$LASTEXITCODE"; exit 1 }

Step 'STEP 2/4 export BVH'
$name = [IO.Path]::GetFileNameWithoutExtension($src)
$pt = Join-Path $GvhmrRepo "outputs\demo\$name\hmr4d_results.pt"
$mocap = Join-Path $Work 'mocap'
& $py (Join-Path $scripts 'gvhmr_export.py') $pt $mocap --tag $tag --fix-root *>> (Join-Path $Work 'gvhmr.log')
if ($LASTEXITCODE -ne 0) { Step "FAILED export exit=$LASTEXITCODE"; exit 1 }

Step 'STEP 3/4 blender render'
$preview = Join-Path $Work 'preview'
New-Item -ItemType Directory -Force -Path (Join-Path $preview 'frames') | Out-Null
$env:BVH_PATH = Join-Path $mocap "$tag.bvh"
$env:NPZ_PATH = Join-Path $mocap "$tag.npz"
$env:OUT_DIR = $preview
$env:PREVIEW_END = '0'
$env:RES = '960x540'
$env:FPS = '30'
$env:CAM_ANGLE = "$CamAngle"
$env:CAM_SIDE = $CamSide
& $blender -b --factory-startup -P (Join-Path $scripts 'bvh_preview_blender.py') *>> (Join-Path $Work 'blender.log')
if ($LASTEXITCODE -ne 0) { Step "FAILED blender exit=$LASTEXITCODE"; exit 1 }

Step 'STEP 4/4 encode mp4s (silent)'
& ffmpeg -v error -y -framerate 30 -i (Join-Path $preview 'frames\f_%04d.png') -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -an (Join-Path $preview "stick_$tag.mp4")
& ffmpeg -v error -y -i $src -i (Join-Path $preview "stick_$tag.mp4") -filter_complex '[0:v]scale=960:540,setsar=1[a];[1:v]scale=960:540,setsar=1[b];[a][b]hstack' -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p (Join-Path $preview "side_by_side_$tag.mp4")
Step 'DONE'