# RAG 知识库问答（rag-knowledge-qa）

基于 **检索增强生成（RAG）** 的企业知识库问答系统：上传 PDF，即可用自然语言提问，
回答**只基于文档内容**并带**可点击的引用溯源**，检索不到就明确拒答，不编造。

技术栈：FastAPI + Qdrant + PostgreSQL + Redis + BGE（embedding/reranker）+ 原生 JS 前端。

![界面](docs/ui.png)

> 截图里的三个能力：中间是流式回答与 `[n]` 引用标注；右侧是命中的原文片段与相关度分数；
> 左侧是知识库文档管理与三种检索模式切换（黄色气泡为"知识库中没有该内容"的拒答）。

---

## 功能

- **PDF 问答**：上传 → 后台解析/清洗/切分/向量化入库，状态实时可见
- **混合检索**：BM25（jieba 中文分词）+ 向量双路召回，RRF 融合名次，再用交叉编码器精排
- **流式输出**：SSE 打字机效果，来源先于正文到达
- **引用溯源**：回答中的 `[1][2]` 可点击，联动高亮右侧对应原文
- **多轮对话**：前端带最近 3 轮历史，后端先做查询改写（指代消解）再检索——
  「它需要哪些依赖」→「运行 RAG agent 需要哪些依赖」；「我不明白」会自动换说法重讲
- **拒答机制**：材料不足时按"是否属于本知识库主题"分三种处理——
  主题内但没写 → 拒答（内部资料不能用外部常识替代）；
  问某个组织的内部事务 → 拒答；完全无关的通用问题 → 用模型自身知识回答，
  并在答案开头标注「（以下不属于知识库内容）」、不标任何引用编号
- **闲聊/身份短路**：`你是谁`、`你好`这类问题不走检索（正则分流，0 成本），
  避免弹出 5 个无关来源误导用户，也省掉一次 embedding+rerank
- **三种检索模式**可切换对比：`vector` / `hybrid` / `hybrid_rerank`
- **评估体系**：检索层命中率 + RAGAS 答案层质量，优化前后全程有数字背书

## 架构

```mermaid
flowchart LR
    A[PDF 上传] --> B[解析 pypdf]
    B --> C[清洗: 去页眉页脚噪声行]
    C --> D[切分 300字/重叠30]
    D --> P[(PostgreSQL chunks<br/>原文 source of truth)]
    P --> E[BGE 向量化]
    E --> F[(Qdrant<br/>向量 + chunk_id)]
    P -. 全量原文 .-> G[BM25 内存索引 jieba]

    Q[用户提问] --> R{检索模式}
    R -->|vector| S[向量检索]
    R -->|hybrid / hybrid_rerank| T[向量 + BM25 双路 top20]
    T --> U[RRF 融合 k=60]
    U --> V{rerank?}
    V -->|是| W[交叉编码器精排 top5<br/>bge-reranker-v2-m3]
    V -->|否| X[top5]
    W --> X
    S --> X
    X --> H[按 chunk_id 回 PG 补原文<br/>BM25 路的已在内存]
    H --> Y{分数低于门槛?}
    Y -->|是| Z[丢弃低分块<br/>材料置空]
    Y -->|否| AA[组装 Prompt: 主题 + 材料 + 问题]
    Z --> AA
    AA --> AB{模型判定}
    AB -->|主题内但材料没写| AC[拒答]
    AB -->|不属于本库的通用问题| AD[通用知识回答<br/>标注"非知识库内容"]
    AB -->|材料够用| AE[带引用回答]
    AD --> AF[SSE: meta → sources → token×N → done]
    AE --> AF
    AC --> AF
```

检索采用工业标准的**两段式架构**：快的海选（向量+BM25 取 top20）→ 准的精排（交叉编码器挑 top5）。

**数据分层**：原文只存 PostgreSQL 一份，Qdrant 里只有向量和一个 `chunk_id`（= chunks 表主键）。
这么切的好处是"向量是派生数据"——换 embedding 模型时原文一个字不动，
跑 `scripts/reindex.py` 重算向量即可，切分结果也没变，所以换模型前后的效果对比是干净的。
反过来（原文只放向量库 payload）时，换模型就得重新解析 PDF、重新切分，
历史结果直接失去可比性；想按文档/页做统计还得全量 scroll。

**为什么门槛前面能"丢材料"却不能"直接拒"**：低分块全是噪声，喂给模型只会干扰；
但"分低"有两种完全不同的原因——库里真没有（该拒），或者这根本不是知识库的问题
（该用模型自己的知识答）。在检索层一刀切拒掉，等于把模型的通用能力阉了。

## 评估

评估集：golden set（有标准答案的题 + 应拒答题），切分参数 300/30。

**题集怎么来的**：`scripts/gen_evalset.py` 让 LLM 只看着单个 chunk 出题
（手写题会不自觉"照着答案出题"，题干和材料高度重合，检索必中，分数虚高），
再用本地向量检索做一遍免费质检（top1 分 + 是否命中标注页），
人工按质检列筛过后 `--finalize` 固化成 `eval/golden_set.json`。
两个评估脚本都默认读这个文件，所以换题集不用改命令。

> 出题时混用了两种风格：一半题不提术语名（考语义检索），一半题直接点名
> 类名/函数名（考关键词检索）。只用一种会让某一检索路占便宜。

> 标注复核（2026-10-26）：原「DeepAgents是什么」标为有答案，复核语料后发现全文
> 只有 URL 路径、代码与导航链接，没有一句定义，已改判为**应拒答**。
> 这条标注错误是在把拒答规则收紧到"定义类问题必须有显式定义句"之后跑出来的——
> 尺子变严，反而先纠正了尺子自己。

> 题集现状（2026-10-26）：已扩到 **29 题**——19 道可答（全部带参考答案）+
> 8 道应拒答（5 主题内缺内容 / 3 组织内部事务）+ 2 道应通用答（与本库无关的
> 通用问题，正确行为是用模型自身知识回答并标注，不是拒答）。
> 下面表格里的数字仍是 **10 题手写基线**，29 题的新数字待重跑后替换。

### 第一层：检索命中率（scripts/eval_retrieval.py，免费秒出）

下面这组数字是 **10 题手写基线**（`--testset legacy` 可复现），保留下来做历史对比：

| 指标 | 纯向量 | 混合(RRF) | 混合+精排 |
|------|:-----:|:---------:|:---------:|
| hit@1 | 43% | 43% | **57%** |
| hit@5 | 57% | 71% | **86%** |
| MRR | 0.476 | 0.548 | **0.679** |
| 拒答正确率 | 0% | 0% | 0% |

精排把 MRR 从 0.55 拉到 0.68：约 1/3 的问题从"正确答案在第 3 名"变成"第 1 条就对"。

**拒答正确率 0% 是这组数字里最有信息量的一列**：3 道应拒答题的 top 分数
（0.545 / 0.605 / 0.661）全部高于 7 道有答案题的最低分（0.438 / 0.508）——
正负例分数区间重叠，**任何单一分数门槛都分不开**。所以拒答最终靠两层：
检索门槛做粗筛（省 token），LLM 按 prompt 判断"材料能否回答"做细判。

### 第二层：答案质量（RAGAS，LLM 当裁判）

| 指标 | 纯向量（6 题） | 混合(RRF)（6 题） | 混合+精排（7 题） |
|------|:-----:|:---------:|:---------:|
| faithfulness（忠实度/反幻觉） | 0.855 | 0.858 | **0.927** |
| answer_relevancy（切题度） | 0.890 | 0.901 | **0.920** |
| context_precision（检索精度） | 0.792 | **0.825** | 0.775 |
| context_recall（检索召回） | **1.000** | 0.722 | 0.810 |

**样本数不同是有原因的**：`怎么创建智能体` 在向量/混合下 top1=0.438 低于门槛被系统拒答
（不计分），而精排分数是另一套标度（0.508 > 0.5），所以被作答了——但它的
context_recall=0（中文"智能体"vs 英文 "agent"，正确页根本没检回来），
单这一题把精排的 recall 拉低了约 0.12。**跨模式比平均数要小心，样本集不一致**。

三个模式各自的胜负点很清楚：

- **faithfulness**：精排最高 0.927。交叉编码器挑出来的块更贴题，模型少发挥
- **context_precision**：混合最高 0.825，精排反而略低——精排偏好代码块，
  把"能回答问题但不是最相关"的块排在了前面
- **context_recall**：纯向量最高 1.000，混合掉到 0.722

recall 的这次反转是**真信号**，不是噪声：逐题看，降分集中在
`如何用LangChain做RAG`（1.0→0.0）、`文本切分器是什么`、`文档切好块之后存到哪`（1.0→0.667）。
原因是**中文提问 × 英文语料**——语料通篇是 LangChain 英文文档，BM25 用中文分词后的
关键词几乎能命中每一块，RRF 融合后把原本正确的块挤出 top5。hit@5 看不出来，
因为它的判定是"沾边就算命中"；RAGAS 的 recall 要求"必要材料真的都在"，一挤就没分。

结论：**混合检索不是普适的赢**。单语场景是纯收益，跨语言场景是"用召回换精度"的赌注。
要根治得换多语言 embedding（BGE-M3），这也解释了为什么 `怎么创建智能体`
在所有模式下都检不回来。

### 性能（RTX 3050 Ti 4GB，本地推理）

| 阶段 | CPU 推理 | GPU 推理 |
|------|:-------:|:-------:|
| 检索（冷启动，一次性） | 28.5s | 25.3s |
| 检索（热态，每次） | 4~12s | **0.49s** |
| 生成（流式） | 3~8s | 3~8s |

## 快速开始

### 0. 前置要求

- Docker Desktop（PostgreSQL / Redis / Qdrant 跑在容器里）
- Python 3.11+（开发环境为 3.13）
- 一个 OpenAI 协议兼容的 LLM API Key（DeepSeek / Qwen / GLM 均可）
- （可选）NVIDIA GPU + CUDA 版 PyTorch，精排提速约 8 倍；CPU 也能跑

### 1. 起依赖服务

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 2. 配置并安装依赖

```bash
cp .env.example .env        # 填入 LLM API Key
pip install -r requirements.txt
```

### 3. 入库文档

```bash
python scripts/ingest.py data/langchain_intro.pdf
```

或者直接在网页里上传 PDF（后台异步入库，状态自动刷新）。

### 4. 启动并访问

```bash
uvicorn app.main:app --reload --port 8000
```

| 入口 | 地址 |
|------|------|
| **网页问答界面** | http://127.0.0.1:8000/ui/ |
| API 文档（Swagger） | http://127.0.0.1:8000/docs |
| 健康检查 | http://127.0.0.1:8000/health |

### 5. 复现评估（可选）

```bash
# 重建向量库：换 embedding 模型后必做（原文在 PG 不动，只重算向量）
python scripts/reindex.py
python scripts/reindex.py --model BAAI/bge-m3   # 指定模型重建（做对比实验）

# 重建题集（可选）：LLM 出题 → 人工筛 eval/evalset_candidates.csv → 固化
python scripts/gen_evalset.py               # 生成候选（约 40 道，几分钱）
python scripts/gen_evalset.py --finalize    # 筛完转成 eval/golden_set.json

python scripts/eval_retrieval.py            # 三种检索模式对比（自动用 golden_set.json）
python scripts/eval_retrieval.py --testset legacy   # 用早期 10 题基线
python scripts/eval_ragas.py                # RAGAS 答案质量(会消耗 LLM API)
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 一次性返回完整 JSON（脚本/评估用） |
| POST | `/chat/stream` | SSE 流式返回（前端用），事件序列 `meta → sources → token×N → done` |
| POST | `/documents/upload` | 上传 PDF，后台异步入库 |
| GET | `/documents` | 文档列表与入库状态 |
| GET | `/health` | PG / Redis / Qdrant 三组件健康检查 |

请求体示例（`history` 可选，最多 20 条，前端只传最近 3 轮）：

```json
{
  "question": "它需要哪些依赖",
  "top_k": 5,
  "mode": "hybrid_rerank",
  "history": [
    { "role": "user", "content": "怎么运行RAG agent" },
    { "role": "assistant", "content": "运行步骤：1 装依赖 2 建向量库 3 初始化并 invoke。" }
  ]
}
```

带 `history` 时后端会先做**查询改写**再检索：把「它需要哪些依赖」改写成
「运行 RAG agent 需要哪些依赖」再去检索（改写结果在 `meta.rewritten_query` 里回传）。
改写只影响检索用的查询，生成时传的仍是用户原话——「我不明白」这类困惑信号不会丢。

## 技术选型与理由

| 决策 | 选择 | 为什么 |
|------|------|--------|
| 前端 | 原生 HTML/CSS/JS | 只有一个页面、两个状态变量，React 的组件复用与状态管理都用不上；引构建链只会把启动从 2 条命令变 6 条。省下的复杂度换成了能讲清原理的 SSE 解析、防 XSS 渲染 |
| 流式 | SSE 而非 WebSocket | 单向推送场景，一个不关闭的 HTTP 响应就够，不需要 WS 的连接管理；且用 `fetch` 读流（而非 EventSource）才能带 JSON body |
| 检索 | 混合 + 精排两段式 | 实测：纯向量 hit@5 57% → 混合 71% → 精排 86%。BM25 补关键词命中，交叉编码器补语义精排 |
| 切分 | 300 字 / 重叠 30 | 消融实验胜出：hit@1 14%→43%，MRR 0.243→0.476（同样 10 题，500/50 对比 300/30） |
| 评估 | 两层而不是 AutoRAG | 10 题 golden set 撑不起网格搜索（会过拟合测试集），借 AutoRAG 的思想（海选用便宜指标、决赛才上 LLM 裁判）而不引框架 |
| Reranker | bge-reranker-v2-m3 | 必须多语言：中文提问 × 英文语料。bge-reranker-base 实测跨语言打分全贴 0.500（sigmoid 零点 = 无判断力），排序等于随机 |

## 踩坑记录（节选）

完整记录见开发笔记，以下是对后来者最有用的几条：

1. **torch 装成 CPU 版**——`torch.cuda.is_available()` 返回 False 时一切推理都在 CPU 硬算；
   同样的 20 对候选，CPU 4s vs GPU 0.42s。装 CUDA 版只需换后缀：`torch==x.y.z+cu132`。
2. **RRF 融合的身份键**——用"原文前 80 字符"做去重键时，跨页重复的代码块会被当成
   同一条，分数重复累加导致排名虚高。改用稳定 id；原文挪进 PG 后直接拿 `chunk_id` 当键。
   改造中又踩了一次：向量路和 BM25 路的键构造方式不一致（一个用 point id、
   另一个用文件名+序号），两路分数合不到一起，混合指标整体下滑 14 个点——
   而纯向量那一路**完全没变**，正是这个对比把问题锁死在融合环节、而不是检索环节。
3. **相关性 ≠ 可回答性**——reranker 只能判断"这段和问题有关"，判断不了"这段能否
   回答问题"。文档通篇讲 LangChain 时，"什么是LangChain"照样拿高分。因此拒答必须
   双保险：分数门槛粗筛 + LLM prompt 细判，实测两层各自独立生效。
4. **切分参数的隐式默认值**——API 上传路径和脚本入库路径用了不同的默认 chunk 参数，
   两条入库路产出不一致。修复方式是把根默认值收口到一处。
5. **静态目录挂载顺序**——`StaticFiles` 挂到 `/` 会成为兜底路由吞掉所有 API，
   必须挂子路径（本项目挂 `/ui`）。
6. **pypdf 抽内联代码会丢字**——PDF 里等宽字体的内联代码是独立文本块，按坐标抽取时
   会被甩到句子后面：
   `Use the RecursiveCharacterTextSplitter to recursively split the documents...`
   抽出来的却是 `Use the to recursively split the documents ... RecursiveCharacterTextSplitter`。
   后果很隐蔽：材料里明明有定义句，模型看到的是残句，按"有没有一句在讲 X 本身"的规则
   判定为没有，于是拒答——而人工看 PDF 原文会觉得"这题明明能答"。
   清洗救不回来（清洗只能删噪声，补不回丢掉的词），要根治得换解析器
   （pdfplumber / pymupdf 能保留文本流顺序）。
7. **评估工具自己也会坏**——生成式题集的质检列一度不可复现：旧代码用
   `recreate_collection` 重建临时集合（新版 qdrant-client 已废弃，行为不保证），
   复算后"19 道可答题全部命中"变成"只命中 7 道"，手工实测确认复算值才对。
   尺子坏了比东西坏了更危险——它会让一份看起来很漂亮的分数完全没有意义。

## 已知限制与后续计划

- [x] chunks 原文入 PostgreSQL（source of truth），Qdrant 只存向量+ID，可随时重建
      （`scripts/reindex.py`；纯向量模式的评估指标改造前后一字不差，说明回填链路正确）
- [x] 评估集从 10 题手写扩到 29 题（从真实 chunk 出题 + 本地质检 + 人工复核）
- [ ] 29 题三模式 + RAGAS 数字重跑，替换掉 10 题基线
- [ ] 标准答案页只标了"出题块所在页"，同一答案在多页重复出现时会低估命中率
      （例：1000 字/200 重叠这组参数第 9 页和第 19 页都写了）。不按 top1 反推着改标注——
      那等于教模型做题，只能人工补标
- [ ] Embedding 换 BGE-M3 对比（bge-small 跨语言弱：中文"智能体"查不到英文 "agent"；
      混合检索在跨语言语料上 context_recall 1.000→0.722，BM25 反而挤掉了正确块）
- [ ] 多文档隔离：无关文档会挤占 top5（实测塞进一份简历后，某题 recall 直接归零），
      需要按知识库/文档做过滤或分集合检索
- [x] 评估集扩充：`gen_evalset.py` 让 LLM 只看着单个 chunk 出题（手写题会"照着答案出题"，
      检索必中、分数虚高），本地向量检索免费质检，人工筛后固化 `eval/golden_set.json`；
      两个评估脚本统一读它，换题集不用改命令
- [ ] 拒答门槛仍靠经验值：正负例分数区间重叠（负例最高 0.661 > 正例最低 0.438），
      分数只做粗筛，真正判定靠 prompt；若要量化拒答质量，需要单独的拒答评估集
- [ ] 通用知识回答没有评估：标注了「以下不属于知识库内容」，但没有指标衡量
      它是否准确；也没接工具，实时信息（天气/股价）只能回答"无法获取"
- [ ] 多轮改写的鲁棒性没有量化：改写错一次，检索就偏一次（当前只在短句/含指代时才调，
      且改写失败自动退回原句）。真要评估需要多轮对话测试集
- [ ] 拒答边界的判定完全依赖 prompt：规则改一行，`什么是LangChain`（应拒）
      和 `文本切分器是什么`（应答）就可能同时翻转。现在靠"正例/反例对照"把两类分开，
      但这是提示词工程的固有脆弱性——更稳的做法是用一个小分类器或独立的可回答性判断

## 目录结构

```
rag-knowledge-qa/
├── app/
│   ├── main.py               # FastAPI 入口(lifespan/CORS/静态挂载)
│   ├── config.py             # 全局配置(pydantic-settings)
│   ├── api/routes/           # chat(问答+SSE流式) / documents / health
│   └── db/                   # SQLAlchemy 异步 ORM + Alembic 迁移
│       └── chunks.py         # chunk 原文读写口(PG 是 source of truth，含同步引擎)
├── rag/                      # RAG 核心
│   ├── parser.py             # PDF 解析
│   ├── cleaner.py            # 噪声行清洗
│   ├── chunker.py            # 切分(300/30)
│   ├── embedder.py           # 向量化(fastembed/bge-small-zh)
│   ├── retriever.py          # Qdrant 向量检索
│   ├── hybrid.py             # BM25 + RRF 融合
│   ├── reranker.py           # 交叉编码器精排(bge-reranker-v2-m3)
│   ├── rewriter.py           # 多轮查询改写(指代消解)
│   ├── smalltalk.py          # 闲聊/身份类短路(不检索直接答)
│   └── generator.py          # Prompt 组装 + 同步/流式生成
├── frontend/                 # 原生单页界面(index.html/style.css/app.js)
├── scripts/                  # ingest / reindex / gen_evalset / eval_* / ask / drop_document
├── docker/docker-compose.yml # PG + Redis + Qdrant + Adminer
├── data/                     # 示例文档
├── docs/                     # 架构说明 + 界面截图
└── eval/                     # 评估数据
```

## License

MIT
