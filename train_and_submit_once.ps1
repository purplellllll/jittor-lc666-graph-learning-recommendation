param(
  [string]$EducoderSessionFile = ".secrets\educoder_session.txt",
  [string]$KernelSlug = "jittor-warmup1-cora-gcn",
  [string]$KaggleJson = "$env:USERPROFILE\.kaggle\credentials.json",
  [string]$KaggleExe = "$env:APPDATA\Python\Python312\Scripts\kaggle.exe"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $EducoderSessionFile)) {
  throw "Educoder session file not found: $EducoderSessionFile"
}

.\run_kaggle_pipeline.ps1 `
  -KernelSlug $KernelSlug `
  -KaggleJson $KaggleJson `
  -KaggleExe $KaggleExe

$session = (Get-Content $EducoderSessionFile -Raw).Trim()
.\educoder_submit.ps1 -EducoderSession $session -ZipPath "result.zip"
