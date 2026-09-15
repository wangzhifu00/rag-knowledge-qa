#!/usr/bin/env bash
# ============================================================
# 验证 RAG 基础环境各服务的连通性
# ============================================================
set -uo pipefail

PASS=0
FAIL=0

check() {
  local label="$1"
  local cmd="$2"

  if eval "$cmd" >/dev/null 2>&1; then
    echo "[ OK ] $label"
    PASS=$((PASS+1))
  else
    echo "[FAIL] $label"
    FAIL=$((FAIL+1))
  fi
}

echo "==> 检查 RAG 基础环境..."
echo ""

# ---------- PostgreSQL ----------
check "PostgreSQL (localhost:5432)" \
  "docker exec rag-postgres pg_isready -U raguser -d ragdb"

# ---------- Redis ----------
check "Redis (localhost:6379)" \
  "docker exec rag-redis redis-cli -a changeme_redis_pwd ping | grep -q PONG"

# 注意:丢弃响应体交给 shell 重定向(check 函数内部已做),
# 不要用 curl 的 -o /dev/null —— Git Bash 下的 curl 是 Windows 原生版,
# 写不了 /dev/null 会返回 exit 23,叠加顶部 pipefail 会把写错误误判成服务失败。

# ---------- Qdrant HTTP ----------
check "Qdrant REST (http://localhost:6333/healthz)" \
  "curl -fsS http://localhost:6333/healthz"

# ---------- Qdrant gRPC 端口监听 ----------
check "Qdrant gRPC (localhost:6334)" \
  "echo > /dev/tcp/localhost/6334"

# ---------- Adminer ----------
# 用 -w 把状态码追加到最后一行再取出来,同样避开 -o /dev/null
check "Adminer (http://localhost:8080)" \
  "curl -fsS -w '\n%{http_code}' http://localhost:8080 | tail -n 1 | grep -q 200"

echo ""
echo "==> 结果: ${PASS} 通过 / ${FAIL} 失败"
echo ""

if [[ $FAIL -gt 0 ]]; then
  echo "[HINT] 查看日志: docker compose logs -f <service>"
  echo "       常见问题:"
  echo "       - 端口被占用: 修改 .env 中的 *_PORT"
  echo "       - 服务未启动: docker compose up -d"
  exit 1
fi

echo "[DONE] 所有服务正常 ✅"
echo ""
echo "下一步:"
echo "  - Adminer:        http://localhost:8080"
echo "  - Qdrant UI:      http://localhost:6333/dashboard"