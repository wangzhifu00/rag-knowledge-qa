"""生成层：把检索到的原文 + 用户问题交给 DeepSeek，组织成带引用的回答。

RAG 的本质在这里：用检索到的材料"限制"LLM 的发挥范围——
它只能根据材料说话，幻觉自然被压下去。
引用格式 [1][2]：逼模型对齐材料，用户也能顺着编号核对原文。
"""

from openai import OpenAI

from app.config import get_settings
from rag.retriever import SearchHit

settings = get_settings()

SYSTEM_PROMPT = (
    "你是一个企业知识库问答助手。规则：\n"
    "1. 只根据下面给出的【参考材料】回答，不要使用材料之外的知识\n"
    "2. 回答中引用某条材料时，在句子末尾标注来源编号，如 [1]、[2]\n"
    "3. 如果材料不足以回答问题，直接说\"根据现有资料无法回答\"，不要编造\n"
    "4. 用简体中文回答，简洁、条理清晰"
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


def format_sources(hits: list[SearchHit]) -> str:
    """把来源格式化成引用清单（命令行/回答末尾都用它）。"""
    lines = []
    for i, hit in enumerate(hits, start=1):
        pages = "、".join(f"第{p}页" for p in hit.page_numbers) or "未知页"
        preview = hit.text[:40].replace("\n", " ")
        lines.append(f"[{i}] {hit.filename}（{pages}） 相似度={hit.score:.3f} | {preview}...")
    return "\n".join(lines)
