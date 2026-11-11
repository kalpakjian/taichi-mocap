# Re-render an existing BVH with another camera angle (no GVHMR re-run).
# Example: .\render_variant.ps1 -Work $HOME\mocap-work -CamAngle 270   # front view
param(
    [string]$Work    = $(if ($env:MOCAP_WORK) { $env:MOCAP_WORK } else { Join-Path $HOME 'mocap-work' }),
    [string]$Tag     = 'gvhmr_full',
    [int]$CamAngle   = 90,
    [string]$CamSide = 'left',
    [string]$Res     = '960x540',
    [int]$Fps        = 30
)
$ErrorActionPreference = 'Continue'
$scripts = Join-Path $PSScriptRoot '..\scripts'
$mocap = Join-Path $Work 'mocap'
$label = "cam$($CamSide.Substring(0,1).ToUpper())$CamAngle"
$out = Join-Path $Work "preview_$label"
$src = Join-Path $Work 'source_30fps.mp4'
$blender = if ($env:BLENDER_EXE) { $env:BLENDER_EXE } else {
    $c = Get-ChildItem (Join-Path $env:ProgramFiles 'Blender Foundation') -Filter blender.exe -Recurse -ErrorAction SilentlyContinue |
         Sort-Object FullName -Descending | Select-Object -First 1
    if ($c) { $c.FullName } else { 'blender' }
}

New-Item -ItemType Directory -Force -Path (Join-Path $out 'frames') | Out-Null
$env:BVH_PATH = Join-Path $mocap "$Tag.bvh"
$env:NPZ_PATH = Join-Path $mocap "$Tag.npz"
$env:OUT_DIR = $out
$env:PREVIEW_END = '0'
$env:RES = $Res
$env:FPS = "$Fps"
$env:CAM_ANGLE = "$CamAngle"
$env:CAM_SIDE = $CamSide

"render -> $out"
& $blender -b --factory-startup -P (Join-Path $scripts 'bvh_preview_blender.py') *>> (Join-Path $Work "blender_$label.log")
if ($LASTEXITCODE -ne 0) { "FAILED blender exit=$LASTEXITCODE"; exit 1 }

& ffmpeg -v error -y -framerate $Fps -i (Join-Path $out 'frames\f_%04d.png') -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -an (Join-Path $out "stick_$label.mp4")
if (Test-Path $src) {
    & ffmpeg -v error -y -i $src -i (Join-Path $out "stick_$label.mp4") -filter_complex '[0:v]scale=960:540,setsar=1[a];[1:v]scale=960:540,setsar=1[b];[a][b]hstack' -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p (Join-Path $out "side_by_side_$label.mp4")
} else { "note: $src not found, skipped the side-by-side encode" }
"done -> $out"