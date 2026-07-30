param(
  [string]$KernelSlug = "jittor-track1-formal-rank-v1",
  [string]$KernelTitle = "Jittor Track1 Formal Rank V1",
  [string]$KaggleJson = "$env:USERPROFILE\.kaggle\credentials.json",
  [string]$KaggleExe = "$env:APPDATA\Python\Python312\Scripts\kaggle.exe",
  [int]$PollSeconds = 60,
  [int]$TimeoutMinutes = 360
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "push_kaggle_track1.ps1") `
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

$finalStatus = & $KaggleExe kernels status $kernel 2>&1 | Out-String
if ($finalStatus -notmatch "complete") {
  throw "Kaggle kernel timed out: $finalStatus"
}

$outputDir = Join-Path $PSScriptRoot "kaggle_output"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
& $KaggleExe kernels output $kernel -p $outputDir --force
if ($LASTEXITCODE -ne 0) {
  throw "Kaggle output download failed with exit code $LASTEXITCODE"
}

$resultPath = Join-Path $outputDir "result.zip"
if (!(Test-Path $resultPath)) {
  throw "Kaggle output did not contain result.zip"
}

Copy-Item -LiteralPath $resultPath -Destination (Join-Path $PSScriptRoot "result.zip") -Force
if (Test-Path (Join-Path $outputDir "outputs\metadata.json")) {
  Copy-Item -LiteralPath (Join-Path $outputDir "outputs\metadata.json") -Destination (Join-Path $PSScriptRoot "metadata.json") -Force
}

Write-Host "Downloaded Kaggle result.zip to formal_track1\result.zip"
