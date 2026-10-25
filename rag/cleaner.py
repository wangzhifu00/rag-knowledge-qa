"""文本清洗：去掉 PDF（网页导出）里的噪声行。

为什么要清：网页导出的 PDF 混着导航栏、页脚、新闻条目和被截断的碎片行，
它们被切成 chunk 入库后，检索时会挤占 top5 名额（B 阶段前的实测：
"Interrupt is coming to NYC" 这种噪声排进了"什么是LangChain"的来源里）。

规则从诊断数据来（78块/735行实测）：
  - 碎片行 <6 字符：134 行
  - 纯 URL 行：30 行
  - 页脚/导航关键词行：Changelog / Forum / Subscribe 等
  - 跨页重复出现的短行（>=3 页）：页眉页脚
清洗要克制：只删"确定是噪声"的，代码行、正文短句都保留——
宁留勿删，删错正文的代价比留下噪声大。
"""

import re
from collections import Counter

from rag.parser import ParsedPage

# 整行只有 URL（可能连着几个）
_URL_ONLY = re.compile(r"^(https?://\S+\s*)+$")

# 常见页脚/导航/营销关键词（命中且行较短才删，防止误伤正文）
_FOOTER_KEYWORDS = (
    "©", "all rights reserved", "cookie", "subscribe", "sign in", "sign up",
    "privacy policy", "terms of service", "was this page helpful",
    "edit this page", "join our", "newsletter", "changelog", "community forum",
    "contact sales", "trust center", "langchain academy",
    "interrupt is coming",  # 本文档的营销横幅（来自实际诊断）
)

# 打印日期行：网页转 PDF 时自动加的页眉，如 "2026/9/20 13:52 Retrieval..."
_DATE_LINE = re.compile(r"^\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2}")

# 纯导航菜单行：1~4 个首字母大写的词、无标点、很短，如
# "Tutorials Deep Agents" / "Resources" / "Careers"。
# 代价：独立成行的短标题（如 "Quickstart"）也会被删——可接受，
# 标题下的正文才是检索主体。
_NAV_MENU = re.compile(
    r"^[A-Z][A-Za-z0-9&+.\-]{1,20}(?:\s+[A-Z][A-Za-z0-9&+.\-]{1,20}){0,3}$"
)

# 跨页重复判定的阈值：出现页数 >= 该值且行较短 → 视为页眉/页脚
_REPEAT_PAGE_THRESHOLD = 3
_REPEAT_MAX_LEN = 50


def _is_junk_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _URL_ONLY.match(s):
        return True
    low = s.lower()
    if len(s) < 150 and any(k in low for k in _FOOTER_KEYWORDS):
        return True
    if len(s) < 6:  # 碎片："D"、"pip i"、"×"
        return True
    if _DATE_LINE.match(s):
        return True
    if len(s) < 40 and _NAV_MENU.match(s):
        return True
    return False


def clean_pages(
    pages: list[ParsedPage],
    drop_repeated: bool = True,
) -> list[ParsedPage]:
    """清洗每个页面的文本行，返回同结构的清洗结果。"""

    # 先统计"跨页重复的短行"——页眉页脚的典型特征
    repeated: set[str] = set()
    if drop_repeated:
        counter: Counter[str] = Counter()
        for p in pages:
            for ln in set(p.text.splitlines()):
                s = ln.strip()
                if s and len(s) < _REPEAT_MAX_LEN:
                    counter[s] += 1
        repeated = {ln for ln, n in counter.items() if n >= _REPEAT_PAGE_THRESHOLD}

    cleaned: list[ParsedPage] = []
    for p in pages:
        kept_lines = []
        for ln in p.text.splitlines():
            s = ln.strip()
            if _is_junk_line(s):
                continue
            if s in repeated:
                continue
            kept_lines.append(s)
        text = "\n".join(kept_lines)
        if text.strip():
            cleaned.append(ParsedPage(page_number=p.page_number, text=text))

    return cleaned
