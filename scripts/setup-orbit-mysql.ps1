<#
Creates Orbit's local MySQL database and least-privilege application account.
Run this locally; it prompts for the MySQL administrator password and never
prints or stores that administrator password in the project.
#>
[CmdletBinding()]
param(
    [string]$DatabaseName = "llm_optimizer",
    [string]$DatabaseUser = "orbit_telemetry"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$mysql = "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $mysql)) { throw "MySQL client not found: $mysql" }
if (-not (Test-Path -LiteralPath $python)) { throw "Project virtual environment not found: $python" }
if ($DatabaseName -notmatch '^[A-Za-z0-9_]+$' -or $DatabaseUser -notmatch '^[A-Za-z0-9_]+$') {
    throw "Database and user names may contain only letters, numbers, and underscores."
}

$service = Get-Service -Name MySQL80 -ErrorAction Stop
if ($service.Status -ne "Running") { throw "MySQL80 must be running before setup." }

$secureRootPassword = Read-Host "Enter the local MySQL root password" -AsSecureString
$rootPassword = [System.Net.NetworkCredential]::new("", $secureRootPassword).Password
$randomBytes = [byte[]]::new(32)
$randomGenerator = [Security.Cryptography.RandomNumberGenerator]::Create()
try {
    # ``Fill`` is unavailable in Windows PowerShell 5.1/.NET Framework.
    $randomGenerator.GetBytes($randomBytes)
}
finally {
    $randomGenerator.Dispose()
}
$appPassword = [Convert]::ToBase64String($randomBytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')

$temporaryDefaults = New-TemporaryFile
$temporarySql = New-TemporaryFile
try {
    [System.IO.File]::WriteAllText($temporaryDefaults, "[client]`nhost=127.0.0.1`nuser=root`npassword=$rootPassword`n", [System.Text.UTF8Encoding]::new($false))
    $sql = @"
CREATE DATABASE IF NOT EXISTS ``$DatabaseName`` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS '$DatabaseUser'@'127.0.0.1' IDENTIFIED BY '$appPassword';
ALTER USER '$DatabaseUser'@'127.0.0.1' IDENTIFIED BY '$appPassword';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX, REFERENCES ON ``$DatabaseName``.* TO '$DatabaseUser'@'127.0.0.1';
FLUSH PRIVILEGES;
"@
    [System.IO.File]::WriteAllText($temporarySql, $sql, [System.Text.UTF8Encoding]::new($false))
    & $mysql "--defaults-extra-file=$temporaryDefaults" --protocol=TCP --execute="source $temporarySql"
    if ($LASTEXITCODE -ne 0) { throw "MySQL database/user setup failed." }
}
finally {
    $rootPassword = $null
    $secureRootPassword = $null
    Remove-Item -LiteralPath $temporaryDefaults, $temporarySql -Force -ErrorAction SilentlyContinue
}

$environmentPath = Join-Path $projectRoot ".env"
$databaseUrl = "DATABASE_URL=mysql+pymysql://$DatabaseUser`:$appPassword@127.0.0.1:3306/$DatabaseName"
$existingLines = if (Test-Path -LiteralPath $environmentPath) { @(Get-Content -LiteralPath $environmentPath) } else { @() }
$updatedLines = @($existingLines | Where-Object { $_ -notmatch '^DATABASE_URL=' }) + $databaseUrl
[System.IO.File]::WriteAllLines($environmentPath, [string[]]$updatedLines, [System.Text.UTF8Encoding]::new($false))

Push-Location $projectRoot
try {
    & $python -m database.initialize
    if ($LASTEXITCODE -ne 0) { throw "The MySQL schema initializer failed." }
}
finally {
    Pop-Location
}

Write-Host "Orbit MySQL telemetry is ready locally. JSONL logging remains enabled." -ForegroundColor Green
