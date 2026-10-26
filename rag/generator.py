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
    "1. 只根据下面给出的【参考材料】回答，不要使用材料之外的知识\n"
    "2. 回答中引用某条材料时，在句子末尾标注来源编号，如 [1]、[2]\n"
    "3. 先判断问题属于哪一类，再按对应标准取舍：\n"
    "   (a) 定义/概念解释类——「X是什么」「什么是X」「解释X」「X的含义」：\n"
    "       材料里必须有一句对 X 本身作出说明的话（形如「X 是……」「X 指……」\n"
    "       「X 用于……」，或对 X 的功能、组成、作用展开描述）。\n"
    "       注意：只出现 X 这个名字不算定义——文件名、URL 路径、import 语句、\n"
    "       目录标题、示例参数里出现 X，都不能当成\"材料解释了 X\"。\n"
    "       不满足时必须以\"根据现有资料无法回答\"开头拒答，说明材料只是提到\n"
    "       该名称而未给出定义，禁止用你自己的知识补全定义\n"
    "   (b) 做法/流程类——「怎么做」「如何」「步骤」「流程」：\n"
    "       材料里有相关步骤、做法或组件即可作答；零散的步骤允许你归纳组织，\n"
    "       不必逐字给出完整教程。（这一类不要拒答）\n"
    "   (c) 列举/事实类——「有哪些」「支持哪些」「存在哪」：\n"
    "       材料里列出的条目就是答案\n"
    "4. 只有当材料确实缺少回答该问题所需的关键信息时才拒答；\n"
    "   材料够用但表述零散时，应归纳整理后回答，不要拒答\n"
    "5. 拒答时必须以\"根据现有资料无法回答\"开头，并说明缺什么，不要编造\n"
    "6. 上面的分类判断是内部动作，绝对不要写进回答里：\n"
    "   不要出现\"这属于做法类问题\"\"根据规则\"\"按第 X 条规定\"这类句子，\n"
    "   直接给答案或直接拒答即可\n"
    "7. 用简体中文回答，简洁、条理清晰"
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


def build_prompt(question: str, hits: list[SearchHit]) -> str:
    """把 top_k 块原文编号拼进 Prompt，模型引用时说 [编号] 即可。"""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        pages = "、".join(f"第{p}页" for p in hit.page_numbers) or "未知页"
        blocks.append(f"[{i}] 来源：{hit.filename}（{pages}）\n{hit.text}")
    context = "\n\n".join(blocks)
    return f"【参考材料】\n{context}\n\n【问题】\n{question}"


def generate_answer(question: str, hits: list[SearchHit]) -> str:
    """调 DeepSeek 生成回答。hits 为空说明检索啥都没找到，直接如实告知。"""
    if not hits:
        return "知识库中没有检索到相关内容，无法回答。"

    client = get_llm()
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        temperature=0.1,  # 知识库问答要"稳"，温度调低减少发挥
    )
    return resp.choices[0].message.content or ""


async def stream_answer(question: str, hits: list[SearchHit]):
    """流式版生成：一段一段吐字，网页那边看起来才像"在说话"。

    和 generate_answer 的区别只有两处：
    1. 客户端换成 AsyncOpenAI（同步版会卡死事件循环）
    2. 传 stream=True，然后 async for 拿增量 chunk.choices[0].delta.content

    最后要 return，不是普通的返回值——这是异步生成器（async yield），
    调用方用 `async for piece in stream_answer(...)` 消费。
    """
    if not hits:
        yield "知识库中没有检索到相关内容，无法回答。"
        return

    client = get_async_llm()
    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
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
