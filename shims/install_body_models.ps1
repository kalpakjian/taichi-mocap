# Put the licensed SMPL-X / SMPL body models where GVHMR expects them.
# Accepts a zip or a loose file found under -SearchDir (default: Downloads + Desktop).
param(
    [string]$GvhmrRepo = $(if ($env:GVHMR_HOME) { $env:GVHMR_HOME } else { Join-Path $HOME 'GVHMR' }),
    [string[]]$SearchDir = @((Join-Path $HOME 'Downloads'), (Join-Path $HOME 'Desktop'))
)
$ErrorActionPreference = 'Continue'
$tmp = Join-Path $env:TEMP 'bodymodels'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$targets = @{
    'SMPLX_NEUTRAL.npz' = @((Join-Path $GvhmrRepo 'inputs\checkpoints\body_models\smplx'), (Join-Path $GvhmrRepo 'inputs\models\smplx'))
    'SMPL_NEUTRAL.pkl'  = @((Join-Path $GvhmrRepo 'inputs\checkpoints\body_models\smpl'))
}
foreach ($dirs in $targets.Values) { foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null } }

# nothing to do if the required model (SMPL-X) is already in place
$done = Test-Path (Join-Path @($targets['SMPLX_NEUTRAL.npz'])[0] 'SMPLX_NEUTRAL.npz')   # SMPL pkl is optional (preview only)
if ($done) { 'SMPLX_NEUTRAL.npz already in place, nothing to do (SMPL_NEUTRAL.pkl is only needed for the mesh preview)'; exit 0 }

# 1. expand any zip that looks like a body-model archive
foreach ($zip in Get-ChildItem -Path $SearchDir -Filter *.zip -ErrorAction SilentlyContinue) {
    $probe = & tar.exe -tf $zip.FullName 2>$null | Select-String -Pattern 'SMPLX|basicModel'
    if ($probe) { "unpacking $($zip.Name)"; Expand-Archive -LiteralPath $zip.FullName -DestinationPath $tmp -Force }
}

# 2. collect candidates
$candidates = Get-ChildItem -Path ($SearchDir + $tmp) -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(SMPLX_NEUTRAL.*\.npz|SMPLX_.*\.npz|basicModel_.*neutral.*\.pkl|SMPL_NEUTRAL.*\.pkl)$' }

$plan = @()
foreach ($c in $candidates) {
    if ($c.Name -match 'SMPLX') { $plan += [pscustomobject]@{ Src = $c.FullName; Name = 'SMPLX_NEUTRAL.npz' } }
    else { $plan += [pscustomobject]@{ Src = $c.FullName; Name = 'SMPL_NEUTRAL.pkl' } }
}
$plan = $plan | Sort-Object Name -Unique

if (-not $plan) {
    'NOT FOUND: no SMPLX_NEUTRAL*.npz / basicModel_*neutral*.pkl under ' + ($SearchDir -join ', ')
    'Download it from https://smpl-x.is.tue.mpg.de/ (Download SMPL-X Model) after clicking the verification email.'
    exit 1
}

foreach ($p in $plan) {
    foreach ($d in $targets[$p.Name]) {
        Copy-Item -LiteralPath $p.Src -Destination (Join-Path $d $p.Name) -Force
        "installed $($p.Name)  <-  $($p.Src)  ->  $d"
    }
}
'done'