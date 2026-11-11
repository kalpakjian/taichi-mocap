$ErrorActionPreference = 'Continue'
$ck = 'C:\Users\princ\GVHMR\inputs\checkpoints'
foreach ($d in 'gvhmr', 'hmr2', 'vitpose', 'yolo') { New-Item -ItemType Directory -Force -Path "$ck\$d" | Out-Null }

$files = @(
    @{ url = 'https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt'; out = "$ck\gvhmr\gvhmr_siga24_release.ckpt"; mb = 156 },
    @{ url = 'https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt'; out = "$ck\yolo\yolov8x.pt"; mb = 131 },
    @{ url = 'https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth'; out = "$ck\vitpose\vitpose-h-multi-coco.pth"; mb = 2431 },
    @{ url = 'https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch=10-step=25000.ckpt'; out = "$ck\hmr2\epoch=10-step=25000.ckpt"; mb = 2584 }
)

foreach ($f in $files) {
    "=== " + $f.out
    if ((Test-Path $f.out) -and ((Get-Item $f.out).Length / 1MB -gt ($f.mb * 0.98))) { "already present"; continue }
    & curl.exe -L --fail --retry 3 --retry-delay 3 -o $f.out $f.url
    if (Test-Path $f.out) { "size MB: " + [math]::Round((Get-Item $f.out).Length / 1MB, 1) }
}
"=== download done ==="
