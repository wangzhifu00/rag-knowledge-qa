# ============================================================
# RAG Knowledge QA —— 应用镜像
#
# 构建：docker compose -f docker/docker-compose.yml build app
# 说明：这是"应用本体"的镜像。PG / Redis / Qdrant 用官方镜像，
#       在 docker/docker-compose.yml 里各自声明，互不干扰。
#
# 三个刻意的决定：
#   1. 模型不进镜像 —— models/ 6.5GB，由 compose 只读挂载到 /app/models。
#      改模型只要换宿主机目录，不用 rebuild（否则每次 rebuild 都在复制 6.5GB）。
#   2. torch 装 CPU 版 —— pypi 在 Linux 上默认给 CUDA 版（2.5GB+），
#      容器里没接 GPU 时那些 .so 全是死重量。CPU 版约 200MB。
#   3. 依赖分两份 —— 容器只装运行期需要的（docker/requirements-api.txt），
#      不装 ragas 那套评估依赖，省下约 1GB。
# ============================================================

FROM python:3.11-slim

# PYTHONUNBUFFERED：日志立刻落盘，否则 docker logs 要缓冲到满才吐
# PYTHONDONTWRITEBYTECODE：容器里不生成 __pycache__，镜像层更干净
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# libgomp1 是 torch 的 OpenMP 运行时，slim 镜像里没有，缺了会报
# "libgomp.so.1: cannot open shared object file"。
# 没装 curl —— 健康检查用 python 自带的 urllib，少一个包少一层。
#
# 先把 Debian 官方源换成阿里云：deb.debian.org 国内实测能让 apt-get update
# 卡上七八分钟，而这一层只需要装一个几百 KB 的包，不值。
# 用 sed 只替换域名，不硬写发行版代号（bookworm / trixie）——基础镜像 tag
# 滚动到下一个 Debian 版本时这行不用改。
# Debian 12+ 的源在 debian.sources（deb822 格式），11 及以前在 sources.list。
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources; \
    else \
      sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list; \
    fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 国内源：pypi 直连经常几百 KB/s，构建能卡十几分钟。
# 用阿里云不用清华 —— 清华的索引和文件偶尔不同步（filelock 4.2.4 索引里有、
# 下载返回 403），阿里云同步完整，实测同一个包能正常下。
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# torch 单独一步、单独索引：必须是 CPU 版，且要在装 sentence-transformers 之前装好，
# 否则后者会把 pypi 的 CUDA 版当依赖重新拉一遍（2.5GB 起）。
#
# 版本为什么要钉死：主索引（阿里云）上的 torch 是无后缀的 CUDA 版，
# 不钉版本时 pip 会跨索引挑"最高版本"，容易把 2.5GB 的 CUDA 包拉下来。
# 钉成 2.14.1+cpu 之后，只有 pytorch 索引能满足这个要求，依赖仍从阿里云拿。
# torch 升级时改这一行的版本号即可。
RUN pip install "torch==2.14.1+cpu" \
      --index-url https://mirrors.aliyun.com/pypi/simple \
      --extra-index-url https://download.pytorch.org/whl/cpu

# 先单独 COPY 依赖清单再装 —— 只改业务代码时这一层走缓存，不用重装依赖
COPY docker/requirements-api.txt ./requirements-api.txt
RUN pip install -r requirements-api.txt

# 业务代码最后进来：它改得最勤，放最后才不会让上面的依赖层反复失效
COPY . .

EXPOSE 8000

# 先跑迁移再起服务：容器可能对着一个旧 schema 的空库启动，
# 少了这步第一个请求就会因为缺表报 UndefinedTable。
# exec 让 uvicorn 接管 PID 1，docker stop 的 SIGTERM 才能直达它（优雅退出）。
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
