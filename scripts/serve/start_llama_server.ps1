# scripts/serve/start_llama_server.ps1
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Server = Join-Path $Root "deployment\llama_cpp\bin\llama-server.exe"
$Model = Join-Path $Root "models\gguf\Qwen3-0.6B-Q8_0.gguf"

if (-not (Test-Path $Server)) {
    throw "Missing llama-server.exe at $Server"
}

if (-not (Test-Path $Model)) {
    throw "Missing GGUF model at $Model"
}

& $Server `
    -m $Model `
    --host 127.0.0.1 `
    --port 8080 `
    --ctx-size 4096 `
    --threads 8
