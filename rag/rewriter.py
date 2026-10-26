"""多轮对话的查询改写（conversational query rewriting）。

为什么必须有这一步：
  检索是"无状态"的——它只看当前这句字符串。而多轮里用户会说
  「它需要哪些依赖」「那这个呢」「我不明白」，这些句子单独拿去检索，
  embedding 和 BM25 都只能抓到"它""这个"这种零信息词，必然检错。

  所以顺序是：先改写 → 再检索 → 最后带着历史生成。
  改写只影响"检索用什么查询"，不改变"回答什么"。

为什么不直接用原始问题 + 历史一起检索：
  向量库里没有"上文"，拼一长串进去只会稀释问题本身的语义。
  正确做法是把指代消解掉，压成一句独立问句。

成本控制：
  · 只在真的有历史时才调（首轮不问，省一次调用）
  · 输出限制在 100 字内，温度 0
  · 模型判断"已经独立"就原样返回，不做无意义的改写
"""

import re

from openai import OpenAI

from app.config import get_settings

settings = get_settings()

REWRITE_SYSTEM_PROMPT = (
    "你是一个检索查询改写器。任务：结合对话历史，把用户最新的一句话改写成"
    "一个不依赖上下文、可以独立拿去检索的完整问题。\n"
    "规则：\n"
    "1. 只输出改写后的问句本身，不要解释、不要加引号、不要加前缀\n"
    "2. 如果最新一句本身已经完整独立（不含\"它/这个/那/上面/刚才/继续\"等指代），"
    "原样输出，不要画蛇添足\n"
    "3. 补全指代时只用历史里出现过的信息，不要自行发挥\n"
    "4. 不超过 100 字"
)


def _format_history(history: list[dict], max_turns: int = 3, clip: int = 200) -> str:
    """把历史整理成短文本：只要最近几轮，助手回复截断（改写不需要全文）。"""
    recent = history[-max_turns * 2:]
    lines = []
    for msg in recent:
        role = "用户" if msg.get("role") == "user" else "助手"
        content = (msg.get("content") or "").strip().replace("\n", " ")
        if len(content) > clip:
            content = content[:clip] + "…"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def needs_rewrite(question: str) -> bool:
    """粗筛：短句或含指代词才值得改写，省掉一次 LLM 调用。

    宁可漏判（漏了就用原句检索，效果退回单轮水平），
    也不要每次都调用——那会给每个带历史的问题白加半秒。
    """
    q = question.strip()
    if len(q) <= 12:
        return True
    pronouns = ("它", "他", "这个", "那个", "这些", "那些", "上面", "刚才",
                "前面", "继续", "再说", "详细点", "举个例子", "为什么")
    return any(p in q for p in pronouns)


# "我不明白"这类是「困惑信号」，不是问题。
# 实测把它们改写成问句是有害的：会丢掉"用户没懂"这层信息，
# 检索又检回同样的材料，模型于是把答案重讲一遍，而不是换个说法解释。
_FEEDBACK = re.compile(
    r"^(我不?明白|我没?懂|不懂|没听懂|没看懂|不太懂|还是不懂|解释一下|"
    r"再说(一遍|清楚)?|详细(一)?点|举个(例子|例)|什么意思|啥意思|然后呢)"
    r"[^\n]{0,8}$"
)


def rewrite_query(question: str, history: list[dict] | None) -> str:
    """返回用于检索的查询。没有历史/不需要改写时原样返回。

    困惑反馈（"我不明白"）用**上一轮的用户提问**去检索——
    直接拿"我不明白"检索，命中的全是无关块（实测 top≈0.50 全噪声）。
    注意改写只换检索用的查询，生成时传的仍是原句，
    "用户在困惑"这个信号不会丢。
    """
    if not history or not needs_rewrite(question):
        return question
    if _FEEDBACK.match(question.strip()):
        prev = next((m["content"] for m in reversed(history)
                     if m.get("role") == "user"), "")
        return prev.strip() or question

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content":
                    f"【对话历史】\n{_format_history(history)}\n\n【最新一句】\n{question}"},
            ],
            temperature=0,
            max_tokens=100,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        # 兜底：模型有时会加引号或写"改写："，清掉；空结果就用原句
        rewritten = rewritten.strip("“”\"'` ")
        for prefix in ("改写后的问题是：", "改写后：", "改写："):
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()
        return rewritten or question
    except Exception:  # noqa: BLE001 - 改写失败不能影响主流程
        # 改写是"增强项"，失败就退回原句检索，绝不能让整个问答挂掉
        return question
