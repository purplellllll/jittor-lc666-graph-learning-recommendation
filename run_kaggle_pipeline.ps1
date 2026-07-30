param(
  [string]$KernelSlug = "jittor-warmup1-cora-gcn",
  [string]$KernelTitle = "Jittor Warmup1 Cora GCN",
  [string]$KaggleJson = "$env:USERPROFILE\.kaggle\credentials.json",
  [string]$KaggleExe = "$env:APPDATA\Python\Python312\Scripts\kaggle.exe",
  [int]$PollSeconds = 60,
  [int]$TimeoutMinutes = 240
)

$ErrorActionPreference = "Stop"

.\push_kaggle.ps1 `
  -KernelSlug $KernelSlug `
  -KernelTitle $KernelTitle `
  -KaggleJson $KaggleJson `
  -KaggleExe $KaggleExe

if (!(Test-Path $KaggleExe)) {
  $cmd = Get-Command kaggle -ErrorAction SilentlyContinue
  if (!$cmd) {
    throw "Kaggle CLI not found. Expected: $KaggleExe"
  }
  $KaggleExe = $cmd.Source
}

$token = Get-Content $KaggleJson -Raw | ConvertFrom-Json
$kernel = "$($token.username)/$KernelSlug"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)

while ((Get-Date) -lt $deadline) {
  $statusText = & $KaggleExe kernels status $kernel 2>&1 | Out-String
  Write-Host $statusText
  if ($statusText -match "complete") {
    break
  }
  if ($statusText -match "error|failed|cancel") {
    throw "Kaggle kernel did not complete: $statusText"
  }
  Start-Sleep -Seconds $PollSeconds
}

New-Item -ItemType Directory -Force -Path kaggle_output | Out-Null
& $KaggleExe kernels output $kernel -p kaggle_output --force

if (!(Test-Path "kaggle_output\result.zip")) {
  throw "Kaggle output did not contain result.zip"
}

Copy-Item "kaggle_output\result.zip" "result.zip" -Force
if (Test-Path "kaggle_output\result.json") {
  Copy-Item "kaggle_output\result.json" "result.json" -Force
}

Write-Host "Downloaded Kaggle result.zip to workspace."
