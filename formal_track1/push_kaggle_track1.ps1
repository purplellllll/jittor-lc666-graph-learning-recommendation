param(
  [string]$KernelSlug = "jittor-track1-formal-rank-v1",
  [string]$KernelTitle = "Jittor Track1 Formal Rank V1",
  [string]$SourceFile = "$PSScriptRoot\track1_solution.py",
  [string]$KaggleJson = "$env:USERPROFILE\.kaggle\credentials.json",
  [string]$KaggleExe = "$env:APPDATA\Python\Python312\Scripts\kaggle.exe",
  [string]$Accelerator = "NvidiaTeslaT4"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $KaggleJson)) {
  throw "Kaggle token not found: $KaggleJson"
}

$token = Get-Content $KaggleJson -Raw | ConvertFrom-Json
if (!$token.username) {
  throw "kaggle credentials do not contain username"
}

if (!(Test-Path $KaggleExe)) {
  $cmd = Get-Command kaggle -ErrorAction SilentlyContinue
  if (!$cmd) {
    throw "Kaggle CLI not found. Expected: $KaggleExe"
  }
  $KaggleExe = $cmd.Source
}

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourcePath = (Resolve-Path $SourceFile).Path
$sourceFullPath = [IO.Path]::GetFullPath($sourcePath)
$workspaceFullPath = [IO.Path]::GetFullPath($workspace)
if (!$sourceFullPath.StartsWith($workspaceFullPath, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to stage source outside workspace: $sourceFullPath"
}
$stageDir = Join-Path $workspace ".kaggle_kernel_formal_track1"
$stageFullPath = [IO.Path]::GetFullPath($stageDir)
if (!$stageFullPath.StartsWith($workspaceFullPath, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to prepare Kaggle staging directory outside workspace: $stageFullPath"
}

if (Test-Path $stageDir) {
  Remove-Item -LiteralPath $stageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $stageDir "track1_solution.py") -Force

$metadata = [ordered]@{
  id = "$($token.username)/$KernelSlug"
  title = $KernelTitle
  code_file = "track1_solution.py"
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

if ($Accelerator) {
  & $KaggleExe kernels push -p $stageDir --accelerator $Accelerator
} else {
  & $KaggleExe kernels push -p $stageDir
}
if ($LASTEXITCODE -ne 0) {
  throw "Kaggle kernel push failed with exit code $LASTEXITCODE"
}

Write-Host "Pushed Kaggle kernel: $($metadata.id)"
