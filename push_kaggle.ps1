param(
  [string]$KernelSlug = "jittor-warmup1-cora-gcn",
  [string]$KernelTitle = "Jittor Warmup1 Cora GCN",
  [string]$KaggleJson = "$env:USERPROFILE\.kaggle\credentials.json",
  [string]$KaggleExe = "$env:APPDATA\Python\Python312\Scripts\kaggle.exe"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $KaggleJson)) {
  throw "Kaggle token not found: $KaggleJson"
}

$token = Get-Content $KaggleJson -Raw | ConvertFrom-Json
if (!$token.username) {
  throw "kaggle.json does not contain username"
}

if (!(Test-Path $KaggleExe)) {
  $cmd = Get-Command kaggle -ErrorAction SilentlyContinue
  if (!$cmd) {
  throw "Kaggle CLI not found. Expected: $KaggleExe"
  }
  $KaggleExe = $cmd.Source
}

python -m pip install -q --upgrade kaggle

$workspace = (Get-Location).Path
$stageDir = Join-Path $workspace ".kaggle_kernel"
$stageFullPath = [IO.Path]::GetFullPath($stageDir)
if (!$stageFullPath.StartsWith([IO.Path]::GetFullPath($workspace), [StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to prepare Kaggle staging directory outside workspace: $stageFullPath"
}
if (Test-Path $stageDir) {
  Remove-Item -LiteralPath $stageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
Copy-Item -LiteralPath "gcn.py" -Destination (Join-Path $stageDir "gcn.py") -Force
if (Test-Path "data\cora.pkl") {
  New-Item -ItemType Directory -Force -Path (Join-Path $stageDir "data") | Out-Null
  Copy-Item -LiteralPath "data\cora.pkl" -Destination (Join-Path $stageDir "data\cora.pkl") -Force
}

$metadata = [ordered]@{
  id = "$($token.username)/$KernelSlug"
  title = $KernelTitle
  code_file = "gcn.py"
  language = "python"
  kernel_type = "script"
  is_private = $true
  enable_gpu = $true
  enable_internet = $true
  dataset_sources = @()
  competition_sources = @()
  kernel_sources = @()
  model_sources = @()
}

$metadataPath = Join-Path $stageDir "kernel-metadata.json"
$metadataJson = $metadata | ConvertTo-Json -Depth 10
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($metadataPath, $metadataJson, $utf8NoBom)

& $KaggleExe kernels push -p $stageDir
if ($LASTEXITCODE -ne 0) {
  throw "Kaggle kernel push failed with exit code $LASTEXITCODE"
}
Write-Host "Pushed Kaggle kernel: $($metadata.id)"
Write-Host "Poll with: $KaggleExe kernels status $($metadata.id)"
Write-Host "Download outputs with: $KaggleExe kernels output $($metadata.id) -p kaggle_output --force"
