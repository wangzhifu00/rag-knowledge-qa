# RAG Knowledge QA

基于 **检索增强生成 (Retrieval-Augmented Generation)** 的知识问答系统骨架。

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| 向量数据库 | **Qdrant** | 文档向量存储与相似度检索 |
| 元数据库 | **PostgreSQL 16** | 用户、会话、文档元信息 |
| 缓存/队列 | **Redis 7** | 检索结果缓存、异步任务队列 |
| LLM | DeepSeek / OpenAI / Qwen / GLM | 答案生成（可切换） |
| Embedding | text-embedding-3-small (默认) | 文档向量化 |
| 可视化 | Adminer | Postgres Web 管理 |

## 目录结构

```
rag-knowledge-qa/
├── docker-compose.yml      # 一键拉起 Postgres + Redis + Qdrant + Adminer
├── .env.example            # 环境变量模板(复制为 .env 并填入 API Key)
├── scripts/
│   ├── start.sh            # 启动所有服务
│   ├── stop.sh             # 停止所有服务
│   └── verify.sh           # 健康检查与连通性验证
├── docs/
│   └── ARCHITECTURE.md     # 架构说明(后续补充)
└── README.md
```

## 快速开始

### 0. 前置条件

- **Docker Desktop for Windows** (含 WSL2 后端): <https://www.docker.com/products/docker-desktop/>
- **Git**: 已安装
- 至少一个 LLM API Key (DeepSeek 推荐,中文场景性价比最高)

### 1. 克隆并配置环境

```bash
git clone <your-repo-url>
cd rag-knowledge-qa

# 复制环境变量模板
cp .env.example .env       # Git Bash / WSL
# PowerShell: Copy-Item .env.example .env

# 编辑 .env,填入至少一个 LLM API Key
notepad .env               # Windows
```

### 2. 启动基础环境

```bash
bash scripts/start.sh
```

或手动:
```bash
docker compose up -d
```

### 3. 验证服务

```bash
bash scripts/verify.sh
```

预期输出:
```
[OK] PostgreSQL    : localhost:5432
[OK] Redis         : localhost:6379
[OK] Qdrant HTTP   : localhost:6333
[OK] Qdrant gRPC   : localhost:6334
[OK] Adminer       : http://localhost:8080
```

### 4. 访问管理界面

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

## License

MIT