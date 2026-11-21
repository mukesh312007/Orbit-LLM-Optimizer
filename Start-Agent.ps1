<# Start the local model and Orbit interface from any PowerShell directory. #>
[CmdletBinding()]
param(
    [string]$Model = "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    [int]$Threads = 4,
    [int]$ContextSize = 1024,
    [int]$ApiPort = 8001,
    [switch]$NoApi
)
$projectRoot = $PSScriptRoot; $python = Join-Path $projectRoot ".venv\Scripts\python.exe"; $modelsRoot = Join-Path $projectRoot "models"; $modelPath = Join-Path $modelsRoot $Model
if (-not (Test-Path -LiteralPath $python)) { throw "Project virtual environment not found: $python" }
if (-not (Test-Path -LiteralPath $modelPath)) { throw "Model not found: $modelPath" }
if (-not ([IO.Path]::GetFullPath($modelPath).StartsWith([IO.Path]::GetFullPath($modelsRoot), [StringComparison]::OrdinalIgnoreCase))) { throw "Model must be located inside the local models directory." }
if (-not (Get-Command llama-server -ErrorAction SilentlyContinue)) { throw "llama-server is not installed or not on PATH." }
function Write-RuntimeState([hashtable]$State) {
    $statePath = Join-Path $projectRoot "runtime\active_runtime.json"
    $json = $State | ConvertTo-Json
    [System.IO.File]::WriteAllText($statePath, $json, [System.Text.UTF8Encoding]::new($false))
}
function Get-ModelId([string]$ModelName) {
    return ([IO.Path]::GetFileNameWithoutExtension($ModelName) -replace '-instruct-', '-')
}
$listening = $false
try { $listening = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:9931/health).StatusCode -eq 200 } catch { }
if (-not $listening) {
    Write-Host "Starting local model: $Model" -ForegroundColor Cyan
    # ArgumentList is deliberately one quoted string: the project path contains spaces.
    $serverArguments = '-m "{0}" --host 127.0.0.1 --port 9931 -t {1} -c {2} -ngl 0' -f $modelPath, $Threads, $ContextSize
    $serverProcess = Start-Process -FilePath "llama-server" -ArgumentList $serverArguments -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddSeconds(90)
    do { Start-Sleep -Milliseconds 700; try { $listening = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:9931/health).StatusCode -eq 200 } catch { $listening = $false } } until ($listening -or (Get-Date) -gt $deadline)
    if (-not $listening) { throw "The local model did not become ready within 90 seconds." }
    $modelId = Get-ModelId $Model
    $runtimeState = @{ model_id = $modelId; model_file = "models/$Model"; threads = $Threads; context_size = $ContextSize; gpu_layers = 0; server_arguments = @(); pid = $serverProcess.Id }
    Write-RuntimeState $runtimeState
}
elseif (-not (Test-Path -LiteralPath (Join-Path $projectRoot "runtime\active_runtime.json"))) {
    # Adopt only a matching local server so Orbit can safely switch it later.
    $matchingServer = Get-CimInstance Win32_Process | Where-Object { $_.Name -like "llama-server*" -and $_.CommandLine -like "*$Model*" } | Select-Object -First 1
    if ($matchingServer) {
        $modelId = Get-ModelId $Model
        Write-RuntimeState @{ model_id = $modelId; model_file = "models/$Model"; threads = $Threads; context_size = $ContextSize; gpu_layers = 0; server_arguments = @(); pid = $matchingServer.ProcessId }
    }
}
elseif ($listening) {
    $matchingServer = Get-CimInstance Win32_Process | Where-Object { $_.Name -like "llama-server*" -and $_.CommandLine -like "*$modelPath*" } | Select-Object -First 1
    if (-not $matchingServer) {
        throw "A llama.cpp server is already listening on port 9931 with a different model. Stop it or select its matching model; Orbit will not terminate an unverified process."
    }
}

function Test-OrbitApi([int]$Port) {
    try {
        $openapi = Invoke-RestMethod -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$Port/openapi.json"
        return $openapi.info.title -eq "Local LLM Runtime Optimizer"
    } catch { return $false }
}

if (-not $NoApi) {
    if (-not (Test-OrbitApi $ApiPort)) {
        $occupied = Get-NetTCPConnection -State Listen -LocalPort $ApiPort -ErrorAction SilentlyContinue
        if ($occupied) { throw "Port $ApiPort is already used by a non-Orbit service. Use -ApiPort with a free loopback port." }
        Start-Process -FilePath $python -ArgumentList @("-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "$ApiPort") -WorkingDirectory $projectRoot -WindowStyle Hidden
        $deadline = (Get-Date).AddSeconds(20)
        do { Start-Sleep -Milliseconds 500 } until ((Test-OrbitApi $ApiPort) -or (Get-Date) -gt $deadline)
        if (-not (Test-OrbitApi $ApiPort)) { throw "Orbit API did not become ready on http://127.0.0.1:$ApiPort." }
    }
    Write-Host "Orbit API is ready at http://127.0.0.1:$ApiPort/docs" -ForegroundColor Green
}
Set-Location $projectRoot
Write-Host "Orbit is ready at http://127.0.0.1:8502" -ForegroundColor Green
& $python -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8502
