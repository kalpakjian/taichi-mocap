# Finds the licence-gated body models the user downloaded (zip or loose file),
# renames them to the names GVHMR/smplx expect, and copies them into place.
#   powershell -ExecutionPolicy Bypass -File install_body_models.ps1
$ErrorActionPreference = 'Continue'
$gv = 'C:\Users\princ\GVHMR'
$targets = @{
    'SMPLX_NEUTRAL.npz' = @("$gv\inputs\checkpoints\body_models\smplx", "$gv\inputs\models\smplx")
    'SMPL_NEUTRAL.pkl'  = @("$gv\inputs\checkpoints\body_models\smpl")
}
foreach ($dirs in $targets.Values) { foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null } }

$searchDirs = @("$env:USERPROFILE\Downloads", "$env:USERPROFILE\Desktop", "$gv\inputs")
$tmp = Join-Path $env:TEMP 'bodymodels'
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# 1. expand any zip that looks like a body-model archive
foreach ($zip in Get-ChildItem -Path $searchDirs -Filter *.zip -ErrorAction SilentlyContinue) {
    $probe = & tar.exe -tf $zip.FullName 2>$null | Select-String -Pattern 'SMPLX|basicModel'
    if ($probe) {
        "unpacking $($zip.Name)"
        Expand-Archive -LiteralPath $zip.FullName -DestinationPath $tmp -Force
    }
}

# 2. collect candidates from Downloads/Desktop/inputs + the unpacked copy
$candidates = Get-ChildItem -Path ($searchDirs + $tmp) -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(SMPLX_NEUTRAL.*\.npz|SMPLX_.*\.npz|basicModel_.*neutral.*\.pkl|SMPL_NEUTRAL.*\.pkl)$' }

$plan = @()
foreach ($c in $candidates) {
    if ($c.Name -match 'SMPLX') { $plan += [pscustomobject]@{ Src = $c.FullName; Name = 'SMPLX_NEUTRAL.npz' } }
    else                        { $plan += [pscustomobject]@{ Src = $c.FullName; Name = 'SMPL_NEUTRAL.pkl' } }
}
$plan = $plan | Sort-Object Name -Unique

if (-not $plan) {
    "NOT FOUND: no SMPLX_NEUTRAL*.npz / basicModel_*neutral*.pkl in Downloads, Desktop or $gv\inputs"
    "Download it from https://smpl-x.is.tue.mpg.de/ (Download SMPL-X Model) after clicking the verification email."
    exit 1
}

foreach ($p in $plan) {
    foreach ($d in $targets[$p.Name]) {
        Copy-Item -LiteralPath $p.Src -Destination (Join-Path $d $p.Name) -Force
        "installed $($p.Name)  <-  $($p.Src)  ->  $d"
    }
}
"done"
