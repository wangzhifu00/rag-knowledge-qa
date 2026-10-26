"""chunk 切分：把整篇文本切成适合检索的小块。

为什么切：1) LLM 上下文有限，整本文档塞不进去
         2) 检索要有粒度——用户问的是小知识点，得能"哪块相关取哪块"

为什么 overlap：一句话跨在两块边界会被切成两句残废，重叠一部分字
保证边界内容在两块里都有完整版本（本项目取 30 字）。

当前实现：按行聚合 + 尾部重叠（baseline）。
路线第 2 周的实验：按标题切 vs 递归切 vs 语义切，到时候换这里。
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    page_numbers: list[int] = field(default_factory=list)  # 涉及的页码（引用溯源用）
    chunk_index: int = 0  # 全文顺序号，排查用


def chunk_pages(
    pages: list,  # list[rag.parser.ParsedPage]
    # 300/30 是 golden 集评估的胜出配置(hit@1 25%→38%, MRR 0.42→0.52)，
    # 作为全项目唯一的根默认值：脚本入库、上传接口都走这里，避免两套参数
    chunk_size: int = 300,
    overlap: int = 30,
) -> list[Chunk]:
    """逐页切分（不跨页合并，保证页码准确）。

    算法：把一页的行按顺序装进"桶"，桶快满(chunk_size)就封口输出；
    封口时把末尾 ~overlap 字的行留下，作为下一个桶的开头（重叠）。
    """
    chunks: list[Chunk] = []
    index = 0

    for page in pages:
        lines = [ln for ln in page.text.splitlines() if ln.strip()]
        current: list[str] = []
        current_len = 0

        for line in lines:
            # 装不下且桶里有东西 → 封口
            if current and current_len + len(line) > chunk_size:
                chunks.append(
                    Chunk(text="\n".join(current),
                          page_numbers=[page.page_number],
                          chunk_index=index)
                )
                index += 1
                # 重叠：从刚封口的桶尾回捞 ~overlap 字的行，作为下一桶开头
                tail: list[str] = []
                tail_len = 0
                for ln in reversed(current):
                    if tail_len + len(ln) > overlap:
                        break
                    tail.insert(0, ln)
                    tail_len += len(ln)
                current, current_len = tail, tail_len

            current.append(line)
            current_len += len(line)

        # 页内最后不满一桶的也输出
        if current:
            chunks.append(
                Chunk(text="\n".join(current),
                      page_numbers=[page.page_number],
                      chunk_index=index)
            )
            index += 1

    return chunks
