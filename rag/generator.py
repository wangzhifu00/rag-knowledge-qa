"""生成层：把检索到的原文 + 用户问题交给 DeepSeek，组织成带引用的回答。

RAG 的本质在这里：用检索到的材料"限制"LLM 的发挥范围——
它只能根据材料说话，幻觉自然被压下去。
引用格式 [1][2]：逼模型对齐材料，用户也能顺着编号核对原文。
"""

from openai import AsyncOpenAI, OpenAI

from app.config import get_settings
from rag.retriever import SearchHit

settings = get_settings()

SYSTEM_PROMPT = (
    "你是一个企业知识库问答助手。规则：\n"
    "1. 优先用【参考材料】回答。用到某条材料时，在句子末尾标注来源编号 [1]、[2]\n"
    "2. 先判断问题属于哪一类（这是你的内部动作，不要写进回答）：\n"
    "   (a) 定义/概念解释类——「X是什么」「什么是X」「解释X」「X的含义」：\n"
    "       判据：材料里有没有一句**在讲 X 本身**的话（X 用来做什么、是什么、\n"
    "       由什么组成）。不是去找固定句式，也不是看 X 这个名字出现了多少次。\n"
    "       正例：材料写着「用 Y 把文档按分隔符切成小块，直到达到指定大小」，\n"
    "       问「Y 是什么」→ 这句就在说明 Y 的用途，可以回答\n"
    "       反例：材料是《用 X 做 RAG 的教程》，通篇在教怎么用 X、装 X、\n"
    "       把 X 的文档切块索引，但没有一句说明 X 是什么 → 问「X 是什么」\n"
    "       必须拒答。因为这种情况下你能答出来，靠的是你自己的常识，\n"
    "       而不是材料——这正是要禁止的\n"
    "       其他只有名字出现的情形（URL 路径、import 语句、目录标题、\n"
    "       示例参数、导航链接）同样一律拒答\n"
    "   (b) 做法/流程类——「怎么做」「如何」「步骤」「流程」：\n"
    "       材料里有相关步骤、做法或组件即可作答；零散的步骤允许你归纳组织，\n"
    "       不必逐字给出完整教程。（这一类不要拒答）\n"
    "   (c) 列举/事实类——「有哪些」「支持哪些」「存在哪」：\n"
    "       材料里列出的条目就是答案\n"
    "3. 材料不足时，按问题是否落在【本知识库主题】范围内分两种处理：\n"
    "   (3-a) 属于本知识库主题、但材料里没讲到 → 必须以\"根据现有资料无法回答\"\n"
    "         开头拒答，并说明缺什么；禁止用你自己的知识补全（内部资料\n"
    "         不能用外部常识替代，否则用户拿到的是似是而非的答案）\n"
    "   (3-b) 问题在问某个具体组织/公司的内部情况（制度、流程、数据、人员、\n"
    "         产品细节），而材料里没有 → 同样拒答，不要用通用知识硬套\n"
    "   (3-c) 与本知识库主题无关的通用问题（编程/技术概念/常识/学习建议等）\n"
    "         → 用你自己的通用知识正常回答，像普通对话一样，但必须做到：\n"
    "         ① 回答第一句加上标记「（以下不属于知识库内容）」\n"
    "         ② 不要给任何 [n] 引用编号\n"
    "         ③ 不确定的地方直说不确定，不要编造\n"
    "4. 拒答时必须以\"根据现有资料无法回答\"开头，并说明缺什么，不要编造\n"
    "5. 上面的分类判断是内部动作，绝对不要写进回答里：\n"
    "   不要出现\"这属于做法类问题\"\"根据规则\"\"按第 X 条规定\"这类句子，\n"
    "   直接给答案或直接拒答即可\n"
    "6. 当问题是在问你自己（如\"你是谁\"\"你是干什么的\"\"你能做什么\"）：\n"
    "   用一句话说明你是基于本知识库文档的问答助手，然后邀请用户提问；\n"
    "   不要把材料里描述的 agent、角色或人格当成你自己\n"
    "7. 如果用户的话不是提问而是一句反馈（如\"我不明白\"\"再说清楚点\"）：\n"
    "   有【对话历史】时，**禁止反问\"你哪里不明白\"，也不要复述用户的\"我不明白\"**，\n"
    "   直接从解释开始：挑上一轮里最可能让人困惑的那个点，换一种说法重讲、\n"
    "   补一个具体例子、拆成更小的步骤，或把抽象描述换成代码/配置示例，\n"
    "   讲完再问一句\"哪部分还需要展开\"即可\n"
    "   完全没有历史时，才说明无法判断所指，请用户补充完整\n"
    "8. 用户的问题可能依赖上一轮（\"它需要哪些依赖\"\"那这个呢\"）：\n"
    "   结合【对话历史】理解指代，但事实内容仍然只从【参考材料】取\n"
    "9. 用简体中文回答，简洁、条理清晰"
)


def get_llm() -> OpenAI:
    """用哪家的模型由 .env 的 LLM_PROVIDER 决定（config.llm_* 已做好切换）。

    DeepSeek / Qwen / GLM 等都兼容 OpenAI 协议，所以统一用 openai 客户端，
    换供应商只改 base_url 和 key，业务代码零改动。
    """
    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def get_async_llm() -> AsyncOpenAI:
    """异步客户端，流式输出(SSE)专用。

    为什么必须用异步版：同步客户端在 async 生成器里一旦发起请求，
    会把整个事件循环卡住——浏览器那边就是一个字都不吐，直到全文生成完。
    异步版可以 async for 逐块拿到增量文本，边收边推。
    """
    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def _format_history(history: list[dict] | None, max_turns: int = 3,
                    clip: int = 300) -> str:
    """对话历史压成文本块。只取最近几轮、助手回复截断——
    生成要的是"上一轮说了什么"，不是全文复读（还省 token）。"""
    if not history:
        return ""
    recent = history[-max_turns * 2:]
    lines = []
    for msg in recent:
        role = "用户" if msg.get("role") == "user" else "助手"
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if len(content) > clip:
            content = content[:clip] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_prompt(question: str, hits: list[SearchHit],
                 history: list[dict] | None = None) -> str:
    """把 top_k 块原文编号拼进 Prompt，模型引用时说 [编号] 即可。

    同时告诉模型"这个知识库讲的是什么"——这是区分两种"答不了"的依据：
      · 话题属于本库但材料没写  → 拒答（不能用常识替代内部资料）
      · 话题根本不属于本库      → 可以用自己的通用知识回答（并标明来源）
    不告诉它主题，模型就分不清这两种情况，只能一律拒答。

    history 非空时插入【对话历史】：让"它需要哪些依赖""我不明白"这类追问
    有上下文可依。注意历史只用于理解指代，事实仍然只从材料取。
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        pages = "、".join(f"第{p}页" for p in hit.page_numbers) or "未知页"
        blocks.append(f"[{i}] 来源：{hit.filename}（{pages}）\n{hit.text}")
    context = "\n\n".join(blocks) or "（本次检索没有命中任何材料）"
    parts = [f"【本知识库主题】\n{settings.kb_description}"]
    history_text = _format_history(history)
    if history_text:
        parts.append(f"【对话历史】\n{history_text}")
    parts.append(f"【参考材料】\n{context}")
    parts.append(f"【问题】\n{question}")
    return "\n\n".join(parts)


def generate_answer(question: str, hits: list[SearchHit],
                    history: list[dict] | None = None) -> str:
    """调 LLM 生成回答。

    hits 为空不再直接返回固定拒答话术：材料为空有两种可能——
    库里真没有（该拒），或者这根本不是知识库的问题（该用通用知识答）。
    这个判断交给 prompt（规则3-a / 3-c），生成层不越权替它决定。
    """
    client = get_llm()
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits, history)},
        ],
        temperature=0.1,  # 知识库问答要"稳"，温度调低减少发挥
    )
    return resp.choices[0].message.content or ""


async def stream_answer(question: str, hits: list[SearchHit],
                        history: list[dict] | None = None):
    """流式版生成：一段一段吐字，网页那边看起来才像"在说话"。

    和 generate_answer 的区别只有两处：
    1. 客户端换成 AsyncOpenAI（同步版会卡死事件循环）
    2. 传 stream=True，然后 async for 拿增量 chunk.choices[0].delta.content

    最后要 return，不是普通的返回值——这是异步生成器（async yield），
    调用方用 `async for piece in stream_answer(...)` 消费。
    hits 为空时不再固定拒答，理由见 generate_answer 的注释。
    """
    client = get_async_llm()
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits, history)},
        ],
        temperature=0.1,
        stream=True,
    )

    async for chunk in stream:
        # 有些 chunk 只带 role/finish_reason，choices 里没有 delta 内容，跳过
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def format_sources(hits: list[SearchHit]) -> str:
    """把来源格式化成引用清单（命令行/回答末尾都用它）。"""
    lines = []
    for i, hit in enumerate(hits, start=1):
        pages = "、".join(f"第{p}页" for p in hit.page_numbers) or "未知页"
        preview = hit.text[:40].replace("\n", " ")
        lines.append(f"[{i}] {hit.filename}（{pages}） 相似度={hit.score:.3f} | {preview}...")
    return "\n".join(lines)
