# Install GVHMR on Windows / Python 3.12, using the pure-torch pytorch3d shim in this folder.
# Override paths with -GvhmrRepo / -Venv, or env GVHMR_HOME / GVHMR_VENV / UV_EXE.
param(
    [string]$GvhmrRepo = $(if ($env:GVHMR_HOME) { $env:GVHMR_HOME } else { Join-Path $HOME 'GVHMR' }),
    [string]$Venv      = $(if ($env:GVHMR_VENV) { $env:GVHMR_VENV } else { Join-Path $HOME 'venvs\gvhmr312' })
)
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$uv = if ($env:UV_EXE) { $env:UV_EXE }
      elseif (Get-Command uv -ErrorAction SilentlyContinue) { (Get-Command uv).Source }
      else { Join-Path $HOME '.local\bin\uv.exe' }
$py = Join-Path $Venv 'Scripts\python.exe'

"=== venv: $Venv"
& $uv venv $Venv --python 3.12
"=== torch 2.3.0 cu121 ==="
& $uv pip install --python $py torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
"=== other deps ==="
& $uv pip install --python $py -r (Join-Path $here 'gvhmr-requirements-win.txt')
"=== hmr4d editable: $GvhmrRepo"
& $uv pip install --python $py --no-deps -e $GvhmrRepo
"=== done ==="