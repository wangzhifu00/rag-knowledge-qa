#!/usr/bin/env bash
# ============================================================
# 停止所有 RAG 基础环境服务(保留数据卷)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "==> 停止所有服务..."
docker compose down

echo ""
echo "[DONE] 服务已停止。数据卷已保留,重启后数据不丢失。"
echo "       如需彻底清理(包括数据): docker compose down -v"