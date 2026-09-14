#!/usr/bin/env bash
# ============================================================
# 启动 RAG 基础环境(Postgres + Redis + Qdrant + Adminer)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 检查 .env 是否存在
if [[ ! -f ".env" ]]; then
  echo "[ERROR] .env 文件不存在,请先:"
  echo "        cp .env.example .env"
  echo "        然后填入 LLM API Key 等配置"
  exit 1
fi

# 检查 docker 是否可用
if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker 未安装。请先安装 Docker Desktop for Windows:"
  echo "        https://www.docker.com/products/docker-desktop/"
  exit 1
fi

echo "==> 拉取镜像..."
docker compose pull

echo "==> 启动服务..."
docker compose up -d

echo "==> 等待服务就绪..."
sleep 5

echo ""
echo "==> 服务状态:"
docker compose ps

echo ""
echo "[DONE] 基础环境已启动。运行 bash scripts/verify.sh 检查连通性"