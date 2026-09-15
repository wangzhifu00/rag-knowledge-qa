<#
.SYNOPSIS
  RAG 基础环境健康检查(Windows PowerShell 版)
#>

$ErrorActionPreference = "Continue"

$pass = 0
$fail = 0

function Test-Service($name, $scriptBlock) {
  if (& $scriptBlock) {
    Write-Host "[ OK ] $name" -ForegroundColor Green
    $script:pass++
  } else {
    Write-Host "[FAIL] $name" -ForegroundColor Red
    $script:fail++
  }
}

Write-Host "==> 检查 RAG 基础环境...`n"

# ---------- PostgreSQL ----------
Test-Service "PostgreSQL (localhost:5432)" {
  docker exec rag-postgres pg_isready -U raguser -d ragdb 2>$null
}

# ---------- Redis ----------
Test-Service "Redis (localhost:6379)" {
  $result = docker exec rag-redis redis-cli -a changeme_redis_pwd --no-auth-warning ping 2>$null
  $result -match "PONG"
}

# ---------- Qdrant HTTP ----------
Test-Service "Qdrant REST (http://localhost:6333/healthz)" {
  try {
    $resp = Invoke-WebRequest "http://localhost:6333/healthz" -UseBasicParsing -TimeoutSec 5
    $resp.StatusCode -eq 200
  } catch { $false }
}

# ---------- Qdrant gRPC 端口 ----------
Test-Service "Qdrant gRPC (localhost:6334)" {
  try {
    $client = New-Object System.Net.Sockets.TcpClient
    $client.BeginConnect("localhost", 6334, $null, $null) | Out-Null
    Start-Sleep -Milliseconds 500
    $client.Connected
  } catch { $false }
}

# ---------- Adminer ----------
Test-Service "Adminer (http://localhost:8080)" {
  try {
    $resp = Invoke-WebRequest "http://localhost:8080" -UseBasicParsing -TimeoutSec 5
    $resp.StatusCode -eq 200
  } catch { $false }
}

Write-Host ""
Write-Host "==> 结果: $pass 通过 / $fail 失败"

if ($fail -gt 0) {
  Write-Host "`n[HINT] 查看日志: docker compose logs -f <service>" -ForegroundColor Yellow
  Write-Host "       常见问题:" -ForegroundColor Yellow
  Write-Host "       - 端口被占用: 修改 .env 中的 *_PORT" -ForegroundColor Yellow
  Write-Host "       - 服务未启动: docker compose up -d" -ForegroundColor Yellow
  exit 1
}

Write-Host "`n[DONE] 所有服务正常 ✅" -ForegroundColor Green
Write-Host ""
Write-Host "下一步:"
Write-Host "  - Adminer:        http://localhost:8080"
Write-Host "  - Qdrant UI:      http://localhost:6333/dashboard"