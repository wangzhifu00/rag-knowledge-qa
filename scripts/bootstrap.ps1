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

# ---------- 1. 检查管理员权限 ----------
Write-Step "检查管理员权限..."
if (-not (New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Write-Fail "需要以管理员身份运行 PowerShell"
  Write-Host "请右键 PowerShell → 以管理员身份运行,然后重跑此脚本" -ForegroundColor Yellow
  pause
  exit 1
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

# ---------- 3. 检查 Docker daemon ----------
Write-Step "检查 Docker daemon..."
try {
  docker ps | Out-Null
  Write-OK "Docker daemon 运行中"
} catch {
  Write-Fail "Docker daemon 未运行。请启动 Docker Desktop,然后重跑本脚本"
  pause
  exit 1
}

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

Write-Step "启动所有服务..."
docker compose up -d

Write-Step "等待服务就绪(10 秒)..."
Start-Sleep -Seconds 10

# ---------- 6. 健康检查 ----------
Write-Step "运行健康检查..."
& "$PSScriptRoot\verify.ps1"

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