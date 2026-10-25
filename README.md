# RAG Knowledge QA

基于 **检索增强生成 (Retrieval-Augmented Generation)** 的知识问答系统骨架。

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| 向量数据库 | **Qdrant** | 文档向量存储与相似度检索 |
| 元数据库 | **PostgreSQL 16** | 用户、会话、文档元信息 |
| 缓存/队列 | **Redis 7** | 检索结果缓存、异步任务队列 |
| LLM | DeepSeek / OpenAI / Qwen / GLM | 答案生成（可切换） |
| Embedding | fastembed（本地 ONNX，当前 `BAAI/bge-small-zh-v1.5`，计划切换 BGE-M3） | 文档向量化 |
| 可视化 | Adminer | Postgres Web 管理 |

## 目录结构

```
rag-knowledge-qa/
├── app/                    # FastAPI 后端(骨架待实现)
├── rag/                    # RAG 核心逻辑(解析、切分、检索、生成,待实现)
├── tests/                  # 测试
├── eval/                   # 评估数据集和脚本
├── docker/
│   └── docker-compose.yml  # 一键拉起 Postgres + Redis + Qdrant + Adminer
├── data/                   # 测试文档(如 langchain_intro.pdf)
├── docs/
│   ├── ARCHITECTURE.md     # 架构说明(后续补充)
│   └── setup-retrospective.html
├── scripts/
│   ├── bootstrap.ps1       # 一键引导(Windows)
│   ├── start.sh / stop.sh  # 启动/停止(有 CRLF 问题,建议用 .ps1)
│   └── verify.sh / verify.ps1  # 健康检查(Windows 用 verify.ps1)
├── requirements.txt        # Python 依赖
├── .env.example            # 环境变量模板(复制为 .env 并填入 API Key)
└── README.md
```

## 快速开始(Windows)

### 0. 一键引导(几乎全自动)

```powershell
# 在项目根目录,右键 PowerShell 以管理员身份运行
cd D:\WorkBuddy_Workspace\2026-09-14-21-55-49\rag-knowledge-qa
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
```

它会自动:
1. 检测管理员权限
2. 没装 Docker 就用 winget 装 Docker Desktop
3. 没建 `.env` 就从模板创建并提示你填 API Key
4. `docker compose pull` + `up -d`
5. 跑健康检查

### 1. 手动路径

如果你想自己控制每一步:

#### 1.1 装 Docker Desktop

> Docker Desktop 是唯一需要手动装的组件(沙箱限制: GUI 安装包 + 需管理员)。

```powershell
# 方式 A: winget 一行(推荐)
winget install --id Docker.DockerDesktop -e --source winget

# 方式 B: 官网下载
# https://www.docker.com/products/docker-desktop/
# 安装时勾选 "Use WSL 2 instead of Hyper-V"(Windows 11 默认)
```

装完**重启电脑**,确认 Docker Desktop 启动并跑出鲸鱼图标,然后 `docker --version` 验证。

#### 1.2 配环境变量

```powershell
Copy-Item .env.example .env
notepad .env    # 填入至少一个 LLM API Key
```

#### 1.3 Python 虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 若 PowerShell 提示"禁止运行脚本"，先执行一次：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

#### 1.4 启动

```powershell
# 一键引导(推荐)
powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

# 或手动(compose 文件在 docker/ 子目录)
docker compose -f docker/docker-compose.yml up -d
```

#### 1.5 验证

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1
```

> Windows 下请用 `.ps1` 版本；`scripts/*.sh` 目前是 CRLF 换行，Git Bash 直接跑会报 `$'\r'` 错误（待修复）。

### 2. 访问管理界面

| 服务 | 地址 | 凭据 |
|------|------|------|
| Adminer (Postgres UI) | <http://localhost:8080> | 用户/密码/库名见 .env |
| Qdrant Dashboard | <http://localhost:6333/dashboard> | 无鉴权(默认) |

## 端口一览

| 端口 | 服务 |
|------|------|
| 5432 | PostgreSQL |
| 6379 | Redis |
| 6333 | Qdrant REST |
| 6334 | Qdrant gRPC |
| 8080 | Adminer (可选) |

## 下一步

- [ ] 接入 LangChain / LlamaIndex / 自研检索链路
- [ ] 文档加载与分块(loader + splitter)
- [ ] Embedding 批处理入库脚本
- [ ] 检索 + Prompt 拼接 + LLM 生成
- [ ] Web UI (Streamlit / Next.js)
- [ ] Embedding 切换 BGE-M3(替换 fastembed 默认的 bge-small-zh-v1.5)
- [ ] 修复 scripts/*.sh 的 CRLF 换行问题(或统一改为 .ps1)
- [ ] 修复 bootstrap.ps1 / start.sh 中裸 docker compose 未指定 -f docker/docker-compose.yml 的问题

## License

MIT
