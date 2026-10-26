# 架构说明

一个单知识库的 RAG 问答服务：上传 PDF → 切分入库 → 混合检索 → 精排 → 带引用生成，
答不出来就拒答。下面是数据怎么流、组件怎么分工、以及每个关键决策的代价。

---

## 1. 系统边界

```
                ┌──────────────────────────────────────────┐
   浏览器 ──────▶│  FastAPI (app/)                          │
   /ui/ 或 /docs │   /chat  /chat/stream  /documents/*      │
                │   /health                                │
                └───────┬──────────────┬───────────────────┘
                        │              │
             ┌──────────▼──────┐  ┌────▼──────────────┐
             │ rag/  检索与生成 │  │ app/db/ 持久层    │
             │  （纯同步，线程池）│  │ （SQLAlchemy 异步）│
             └───┬────────┬────┘  └────┬──────────────┘
                 │        │            │
        ┌────────▼──┐  ┌──▼─────┐  ┌───▼──────────┐
        │  Qdrant   │  │  PG    │  │  Redis       │
        │ 只存向量+ID│  │ 原文+元数据│  │ （预留）      │
        └───────────┘  └────────┘  └──────────────┘
                 │
        ┌────────▼─────────────────────────┐
        │ 本地模型（models/，不进镜像）        │
        │ bge-m3 (embedding) + bge-reranker │
        │  + jieba (BM25 分词)              │
        └──────────────────────────────────┘
```

外部依赖只有一个：任何 OpenAI 协议兼容的 LLM 接口（当前配的是阿里云百炼的
`qwen-plus-2025-12-01`）。**embedding 和 rerank 全在本地跑**，不出网、不花钱、不依赖额度。

---

## 2. 数据流 A：入库（离线）

```
PDF ──parser──▶ 每页文本 ──cleaner──▶ 去噪行 ──chunker──▶ 块 ──embedder──▶ 向量
                    │                                        │
                    └────────── 原文 ────────────────────────┴──▶ PG(chunks)
                                                                  │
                                                          向量 ──▶ Qdrant
```

| 阶段 | 位置 | 关键点 |
|---|---|---|
| 解析 | `rag/parser.py` | pypdf 逐页抽文本，保留页号（引用要标页）。**会丢内联代码的词**，见 README 坑 6 |
| 清洗 | `rag/cleaner.py` | 去页眉页脚、孤立页码、重复空行。只能删噪声，补不回解析阶段丢掉的词 |
| 切分 | `rag/chunker.py` | **按行装桶**，`chunk_size=300 / overlap=30`。按行而不是按字符，是为了不让一句话被拦腰切开。这组参数是消融出来的（500/50 对比下 hit@1 14%→43%） |
| 向量化 | `rag/embedder.py` | bge-m3，1024 维，fp16 |
| 落库 | `rag/retriever.py` + `app/db/chunks.py` | **原文进 PG，Qdrant 的 payload 只放 `{"chunk_id": id}`** |

两个入口共用同一条链路：`scripts/ingest.py`（命令行）和
`POST /documents/upload`（网页，`asyncio.to_thread` 后台跑，状态可轮询）。

---

## 3. 数据流 B：问答（在线）

```
问题
 │
 ├─(指代/短句)─▶ rewriter 改写 ─┐
 │                             ▼
 ├──────────────────────▶ 双路召回 ─────────────┐
 │                        │                    │
 │                  BM25 (jieba)          Qdrant 向量
 │                  各取 depth=20               │
 │                        └──────┬─────────────┘
 │                               ▼
 │                        RRF 融合 (k=60)
 │                               ▼
 │                        reranker 精排 ──▶ top5
 │                               ▼
 │                        前置拒答判定 ──┬─ 拒答 ─▶ 固定话术（sources 置空）
 │                                       └─ 通过 ─┐
 │                                                ▼
 └──────────────────────────────────────▶ generator 组装 Prompt
                                                  │
                                         同步 / SSE 流式
                                                  ▼
                                          引用编号兜底 ──▶ 响应
```

四个值得单独说的点：

**① 混合检索用 RRF 而不是分数加权。**
BM25 给的是无上界的正数，向量给的是 cosine，两者量纲不可比，加权求和还要人调权重。
RRF 只看名次：`score = Σ 1/(k + rank)`，`k=60`。代价是丢掉分数里的"置信度"信息，
所以拒答门槛不能从 RRF 分数取（见下一条）。

**② 拒答门槛取自精排概率，不取自融合分数。**
bge-m3 的 cosine 在正负例上区间**重叠**——"材料沾边但答不了"的题能拿到比
"真正可答"的题还高的分，任何单一阈值都分不开（具体数值见 README 第一层评估）。
reranker 输出的 logits 经过 sigmoid 后区分度高得多，所以
`hybrid_rerank` 模式用 reranker 概率当主门槛，其余模式退回 cosine 粗筛。
**分数只做粗筛，真正的判定靠 prompt。**

**③ 拒答是"前置"的，不是靠模型自己收敛。**
检索后先判一次，低于门槛直接返回固定话术并把 `sources` 置空，不进生成。
理由：让 LLM 自己说"我不知道"不可靠，而且浪费一次调用。

**④ 引用编号有兜底。**
模型偶尔会引用不存在的编号（`[7]` 但只有 5 条来源）。`_clamp_citations`
把越界编号剥掉，并且**跳过 ``` 围栏内的代码**、
**跳过英文标识符后面的方括号**（`chunks[0]` 是数组下标，不是引用）。

---

## 4. 数据契约

### PostgreSQL

```
documents(id, filename UNIQUE, file_path, file_size, mime_type,
          status, error_message, created_at, updated_at)
    │ 1:N  ON DELETE CASCADE
chunks(id, document_id, chunk_index, text, page_numbers[], char_count, created_at)
    UNIQUE(document_id, chunk_index)
```

`status` 是入库状态机：`pending → processing → ready / failed`，网页靠轮询它显示进度。

### Qdrant

```
collection: documents_chunks
point: id = chunk_id, vector = 1024d, payload = {"chunk_id": <int>}
```

**Qdrant 里没有原文、没有文件名。** 命中之后 `hydrate()` 拿 `chunk_id` 回 PG 取全文。
这不省那点存储，换来的是"删掉向量库可以无损重建"（`scripts/reindex.py`，105 块 1.2 秒）。
评估脚本另建临时集合，那种集合的 payload 里带 `text`，由 `chunk_index == -1` 区分。

### 评估数据

`eval/golden_set.json`：每题记 `question / expected(verdict) / answer / source_pages`。
`expected` 取值 `answerable | refuse | general`，三个评估脚本共读这一份，
换题集不用改命令。

---

## 5. 设备与精度

`rag/device.py` 统一决定 device 和 dtype：

- `EMBEDDING_DEVICE=auto`：有 CUDA 就用 `cuda`，否则 `cpu`
- `GPU_FP16=true`：`model_kwargs={"torch_dtype": float16}` **必须传在
  `from_pretrained` 阶段**，加载后再转是无效的
- reranker 的 `batch_size=8`（不是默认 32）：4GB 显存下 32 对会 OOM

两个 568M 模型 fp16 后峰值 2.12GB / 4.00GB。容器里跑 CPU，所以
`docker/requirements-api.txt` 装的是 CPU 版 torch。

---

## 6. 关键决策与代价

| 决策 | 为什么 | 代价 |
|---|---|---|
| 原文进 PG，Qdrant 只存向量 | 向量库变可重建的派生数据，换模型不用重新解析 PDF | 每次检索多一次 PG 查询 |
| embedding 用 bge-m3 而不是 bge-small | 中文提问 × 英文语料时 bge-small 跨语言弱（纯向量 hit@5 59% → 88%） | 启动加载 22 秒，显存约 1.15GB |
| 前置拒答 + prompt 双向规则 | 让模型自己收敛不可靠；规则成对写（该拒的拒、该答的答）才不会互相带偏 | 规则改一行，边界两侧可能同时翻转，靠 29 题回归守 |
| 混合检索默认开精排 | RAGAS context_precision 0.789 → 0.862 | 每次多 1~2 秒 |
| 评估拆成三层（检索/拒答/RAGAS） | 成本差两个数量级，便宜的先跑能定位问题是检索还是生成 | 需要维护两套题集（检索用页号、拒答用 verdict） |
| 模型不进镜像 | 6.5GB，改模型不该触发 rebuild | clone 后必须先下模型，容器才真正可用 |

---

## 7. 扩展点

**换生成模型**：改 `.env` 的 `OPENAI_MODEL` / `OPENAI_API_KEY` / `OPENAI_BASE_URL`
（key 和 base_url 必须成对，否则 401）。换完跑一次 `scripts/eval_boundary.py`
（3 分钟、5 万 token），确认"该答/该拒"这条线没移动。

**换 embedding**：改 `EMBEDDING_MODEL_NAME` → `python scripts/reindex.py` 重建向量
（原文在 PG，不用重新解析）→ 所有评估数字重跑。

**换解析器**：只动 `rag/parser.py` 的输出契约（`list[(page_no, text)]`），
下游不变。注意换完要重跑评估集，因为 pypdf 的丢字问题可能"遮挡"了某些错标。

**加文档**：`POST /documents/upload` 或 `scripts/ingest.py`。
当前是**单知识库**，多文档之间不做隔离，无关文档会挤占 top5 —— 见 README 已知限制。

**加评估题**：`scripts/gen_evalset.py`（LLM 只看单个 chunk 出题 → 本地质检 → 人工筛）
→ `--finalize` 写入 `golden_set.json`。手写题容易"照着答案出题"，检索必中、分数虚高。

---

## 8. 部署形态

| 形态 | 命令 | 用途 |
|---|---|---|
| 容器全栈 | `docker compose -f docker/docker-compose.yml up -d --build` | 演示 / 交付。app 与 PG/Redis/Qdrant/Adminer 同一网络，容器内用服务名互访 |
| 本地开发 | `uvicorn app.main:app --reload --port 8000` | 调参 / 跑评估。评估脚本依赖 ragas，只在本地 venv 装 |

两者共用同一套数据卷，切换不影响数据。容器内 `POSTGRES_HOST` / `QDRANT_HOST`
被 compose 覆盖成服务名 —— 容器里的 `localhost` 指的是容器自己，不是宿主机。

容器启动顺序：`alembic upgrade head` → `uvicorn`（`exec` 让 uvicorn 接管 PID 1，
`docker stop` 的 SIGTERM 才能直达它）→ lifespan 里预热模型和 BM25。

---

更细的指标、踩坑记录和已知短板见根目录 `README.md`。
