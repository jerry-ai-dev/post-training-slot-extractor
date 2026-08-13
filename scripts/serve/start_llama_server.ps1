param(
    [Parameter(Mandatory = $true)][string]$ModelId,
    [int]$Port = 8080,
    [int]$Threads = 8
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Server = Join-Path $Root "deployment\llama_cpp\llama-server.exe"

if (-not (Test-Path $Server)) {
    throw "Missing llama-server.exe at $Server"
}

Push-Location $Root
try {
    uv run python -m slot_extractor.inference.llama_server_manager `
        --server $Server --model-id $ModelId --port $Port --threads $Threads
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
