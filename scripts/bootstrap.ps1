<#
.SYNOPSIS
  RAG Knowledge QA - 一键引导脚本
.DESCRIPTION
  检测环境 → 安装 Docker Desktop(如缺失) → 启动 Postgres + Redis + Qdrant → 健康检查
.NOTES
  需要管理员权限运行(UAC 弹窗)
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# ---------- 彩色输出 ----------
function Write-Step($msg)   { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-OK($msg)     { Write-Host "[ OK ] $msg" -ForegroundColor Green }
function Write-Fail($msg)   { Write-Host "[FAIL] $msg" -ForegroundColor Red }
function Write-Warn($msg)   { Write-Host "[WARN] $msg" -ForegroundColor Yellow }

# ---------- 1. 管理员权限(缺失则自动提权重启) ----------
Write-Step "检查管理员权限..."
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
  ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
  Write-Warn "当前非管理员权限,正在弹出 UAC 请求提权..."
  Start-Process powershell -Verb RunAs -ArgumentList "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`""
  exit 0
}
Write-OK "管理员权限"

# ---------- 2. 检查 / 安装 Docker Desktop ----------
Write-Step "检查 Docker Desktop..."
$dockerInstalled = $false

if (Get-Command docker -ErrorAction SilentlyContinue) {
  $dockerInstalled = $true
  $version = docker --version 2>$null
  Write-OK "Docker 已安装: $version"
}

if (-not $dockerInstalled) {
  Write-Warn "Docker Desktop 未安装"

  if (Get-Command winget -ErrorAction SilentlyContinue) {
    $choice = Read-Host "用 winget 一键安装 Docker Desktop? (Y/n)"
    if ($choice -eq "" -or $choice -match "^[Yy]") {
      Write-Step "通过 winget 安装 Docker Desktop(约 500MB,需 2-5 分钟)..."
      winget install --id Docker.DockerDesktop -e --source winget --accept-package-agreements --accept-source-agreements
      Write-OK "安装完成"
      Write-Warn "请重启 PowerShell 让 docker 命令生效,然后重跑本脚本"
      exit 0
    }
  }

  Write-Host "`n手动安装步骤:" -ForegroundColor Yellow
  Write-Host "  1. 访问 https://www.docker.com/products/docker-desktop/"
  Write-Host "  2. 下载 Docker Desktop Installer.exe"
  Write-Host "  3. 安装时勾选 WSL2 后端"
  Write-Host "  4. 重启电脑"
  Write-Host "  5. 重跑本脚本`n"
  pause
  exit 1
}

# ---------- 3. 检查 / 自动启动 Docker daemon ----------
Write-Step "检查 Docker daemon..."

function Test-DockerDaemon {
  docker ps 2>&1 | Out-Null
  return ($LASTEXITCODE -eq 0)
}

if (-not (Test-DockerDaemon)) {
  $dockerExe = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
  if (Test-Path $dockerExe) {
    Write-Warn "Docker daemon 未运行，正在启动 Docker Desktop..."
    Start-Process $dockerExe
    $elapsed = 0
    while ($elapsed -lt 90 -and -not (Test-DockerDaemon)) {
      Start-Sleep -Seconds 3
      $elapsed += 3
      Write-Host "  等待 Docker daemon 就绪... ($elapsed 秒)"
    }
  }
}

if (-not (Test-DockerDaemon)) {
  Write-Fail "Docker daemon 未运行。请手动启动 Docker Desktop，然后重跑本脚本"
  pause
  exit 1
}
Write-OK "Docker daemon 运行中"

# ---------- 4. 检查 .env ----------
Set-Location $ProjectRoot
Write-Step "检查 .env 配置..."
if (-not (Test-Path ".env")) {
  Copy-Item ".env.example" ".env"
  Write-Warn ".env 已从模板创建,请填入至少一个 LLM API Key"
  notepad.exe .env
  $choice = Read-Host "填完 API Key 后按 Enter 继续,或输入 n 跳过(仅启动基础服务)"
  if ($choice -match "^[Nn]") {
    Write-Warn "跳过 LLM 配置,继续启动基础服务"
  }
}

# ---------- 5. 拉镜像 + 启动 ----------
Write-Step "拉取 Docker 镜像(Postgres + Redis + Qdrant)..."
docker compose pull
if ($LASTEXITCODE -ne 0) {
  Write-Fail "镜像拉取失败，可能是网络问题。请为 Docker Desktop 配置镜像加速器后重跑脚本"
  Write-Host "参考: 设置 → Docker Engine → registry-mirrors，添加国内镜像地址" -ForegroundColor Yellow
  exit 1
}

Write-Step "启动所有服务..."
docker compose up -d
if ($LASTEXITCODE -ne 0) {
  Write-Fail "docker compose up 启动失败"
  exit 1
}

Write-Step "等待服务就绪(10 秒)..."
Start-Sleep -Seconds 10

# ---------- 6. 健康检查 ----------
Write-Step "运行健康检查..."
& "$PSScriptRoot\verify.ps1"
$verifyExit = $LASTEXITCODE

if ($verifyExit -ne 0) {
  Write-Fail "健康检查未全部通过，请根据上方 FAIL 项排查"
  exit 1
}

Write-Host "`n============================================="
Write-OK "RAG 基础环境已就绪!"
Write-Host "============================================="
Write-Host ""
Write-Host "访问入口:"
Write-Host "  Adminer (DB UI)     : http://localhost:8080"
Write-Host "  Qdrant Dashboard    : http://localhost:6333/dashboard"
Write-Host ""
Write-Host "常用命令:"
Write-Host "  停止服务            : docker compose down"
Write-Host "  查看日志            : docker compose logs -f"
Write-Host "  重启单个服务        : docker compose restart postgres"
Write-Host ""
Write-Host "下一步:接入 LangChain / LlamaIndex,开始构建 RAG 链路"
Write-Host ""