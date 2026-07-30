param(
  [Parameter(Mandatory = $true)]
  [string]$EducoderSession,

  [string]$ZipPath = "result.zip",
  [string]$Identifier = "Jittor-7"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ZipPath)) {
  throw "Submission zip not found: $ZipPath"
}

$ak = "e9dd5b4322f9f7d83d009de9bfa100c3"
$sk = "2e3da06ae26ba9f76a5d8d355746f2fe"

function New-EduHeaders([string]$method) {
  $ts = [DateTimeOffset]::Now.ToUnixTimeMilliseconds()
  $sigSource = "method=$($method.ToUpper())&ak=$ak&sk=$sk&time=$ts"
  $b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($sigSource))
  $md5 = [Security.Cryptography.MD5]::Create()
  $sig = ($md5.ComputeHash([Text.Encoding]::UTF8.GetBytes($b64)) | ForEach-Object { $_.ToString("x2") }) -join ""
  return @{
    "Accept" = "application/json"
    "Origin" = "https://www.educoder.net"
    "Cookie" = "_educoder_session=$EducoderSession"
    "Pc-Authorization" = $EducoderSession
    "X-EDU-Type" = "pc"
    "X-EDU-Timestamp" = "$ts"
    "X-EDU-Signature" = $sig
    "X-Original-Protocol" = "https:"
    "X-Original-Host" = "www.educoder.net"
    "X-Original-Origin" = "https://www.educoder.net"
    "X-Request-Id" = [guid]::NewGuid().ToString()
  }
}

Write-Host "Uploading $ZipPath ..."
$uploadHeaders = New-EduHeaders "POST"
$curlArgs = @(
  "-sS",
  "-X", "POST",
  "https://data.educoder.net/api/attachments.json",
  "-F", "file=@$((Get-Item $ZipPath).FullName)"
)
foreach ($key in $uploadHeaders.Keys) {
  $curlArgs += @("-H", "$key`: $($uploadHeaders[$key])")
}
$uploadText = & curl.exe @curlArgs
if ($LASTEXITCODE -ne 0) {
  throw "curl upload failed with exit code $LASTEXITCODE"
}
$upload = $uploadText | ConvertFrom-Json

$attachmentId = $upload.id
if (!$attachmentId) {
  throw "Upload failed: $($upload | ConvertTo-Json -Depth 20)"
}
Write-Host "Uploaded attachment_id=$attachmentId"

Write-Host "Submitting attachment to competition..."
$submitHeaders = New-EduHeaders "POST"
$submitHeaders["Content-Type"] = "application/json; charset=utf-8"
$body = @{
  attachment_id = $attachmentId
  upload_file_url = ""
} | ConvertTo-Json -Compress

$submit = Invoke-RestMethod `
  -Uri "https://data.educoder.net/api/competitions/$Identifier/upload_file.json" `
  -Method Post `
  -Headers $submitHeaders `
  -Body $body

Write-Host ($submit | ConvertTo-Json -Depth 20)

Write-Host "Fetching your visible submit records..."
$recordsHeaders = New-EduHeaders "GET"
$records = Invoke-RestMethod `
  -Uri "https://data.educoder.net/api/competitions/$Identifier/results.json?stage_id=578" `
  -Method Get `
  -Headers $recordsHeaders
Write-Host ($records | ConvertTo-Json -Depth 20)
