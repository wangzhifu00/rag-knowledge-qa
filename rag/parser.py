"""PDF 解析：把 PDF 变成带页码的纯文本。
为什么保留页码：它是以后"引用溯源"的坐标——回答里能写
(langchain_intro.pdf, 第12页)，用户才能核对。没有页码的引用是假的。
"""

from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader

@dataclass
class ParsedPage:
    page_number: int  # 从 1 开始，和 PDF 阅读器里看到的页码一致
    text: str

def _normalize(text: str) -> str:
    """轻度清洗：压缩连续空格、去掉行尾空白。不删换行（切分时按行聚合）。"""
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())  # 压缩行内连续空白
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_pdf(path: str | Path) -> list[ParsedPage]:
    """逐页提取文本。空页跳过，但页码保持真实（不连续也没关系）。"""
    reader = PdfReader(str(path))
    pages: list[ParsedPage] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _normalize(text)
        if text:
            pages.append(ParsedPage(page_number=i, text=text))
    if not pages:
        raise ValueError(f"{path} 里没有提取到任何文本（可能是扫描版 PDF，需要 OCR）")
    return pages
