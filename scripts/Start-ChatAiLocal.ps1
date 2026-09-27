param(
    [Parameter(Mandatory)][ValidateSet('Development')][string]$Environment,
    [Parameter(Mandatory)][string]$ApiKeyFile,
    [Parameter(Mandatory)][string]$EnvironmentFile,
    [ValidateRange(1024, 65535)][int]$Port = 8000,
    [switch]$DisableWarmup,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$aiRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $aiRoot '.venv/Scripts/python.exe'

# 기존 가상 환경을 사용하고 비밀 원문은 명령행이나 부모 process 환경에 넣지 않는다.
foreach ($path in @($ApiKeyFile, $EnvironmentFile)) {
    if (-not [IO.Path]::IsPathFullyQualified($path) -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw 'AI local launch requires existing absolute configuration and secret-file paths.'
    }
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'The existing AI Python virtual environment is required; no package installation was attempted.'
}

$arguments = @(
    (Join-Path $PSScriptRoot 'chat_ai_local.py'),
    '--environment', $Environment,
    '--api-key-file', $ApiKeyFile,
    '--environment-file', $EnvironmentFile,
    '--port', $Port.ToString()
)
if ($DisableWarmup) { $arguments += '--disable-warmup' }
if ($ValidateOnly) { $arguments += '--validate-only' }

# 자식만 설정을 공급받는다. 성공/실패에도 호출자의 환경과 기존 .env는 바뀌지 않는다.
Push-Location -LiteralPath $aiRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "AI local launch failed with exit code $LASTEXITCODE." }
} finally {
    Pop-Location
}
