$ErrorActionPreference = 'Continue'
$uv = 'C:\Users\princ\.local\bin\uv.exe'
$venv = 'C:\Users\princ\venvs\gvhmr312'
$py = "$venv\Scripts\python.exe"

"=== venv ==="
& $uv venv $venv --python 3.12
"=== torch 2.3.0 cu121 ==="
& $uv pip install --python $py torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
"=== other deps ==="
& $uv pip install --python $py -r 'C:\Users\princ\shims\gvhmr-requirements-win.txt'
"=== hmr4d editable ==="
& $uv pip install --python $py --no-deps -e 'C:\Users\princ\GVHMR'
"=== done ==="
