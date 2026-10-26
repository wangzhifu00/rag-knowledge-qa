"""评估集生成器：让 LLM 从真实语料里出题 → 人工筛 → 成为正式 golden 集。

为什么不再手写题目：
  手写题会不自觉地"照着答案出题"——题干里的措辞和材料高度重合，
  关键词检索必然命中，量出来的分数虚高。让 LLM 只看到单个 chunk 出题，
  题干更接近真实用户，难度也更接近线上。

为什么必须人工筛：
  LLM 出的题会出现三类问题：① 问的东西材料其实没讲清（该拒答却标了可答）
  ② 题干太泛（"这份文档讲了什么"）③ 和别的题重复。
  脚本已经用本地向量检索做了一遍**免费质检**，把 top1 分数和是否命中
  都写进 CSV 了，人工只需照着这两列重点看可疑项。

流程：
  1. generate（默认）：解析 → 清洗 → 切分 → 按页轮转抽块 → LLM 出题
     → 本地质检 → eval/evalset_candidates.csv（Excel 直接打开，不乱码）
  2. 人工筛：删掉不要的行，或在「保留」列填 n
  3. finalize：读回 CSV → eval/golden_set.json（评估脚本的正式题源）

用法（项目根目录）：
    python scripts/gen_evalset.py                 # 生成 ~40 道候选题
    python scripts/gen_evalset.py --n 30          # 指定可答题数量
    python scripts/gen_evalset.py --finalize      # 人工筛完，转成正式题集

成本：每块一次 LLM 调用（40 次左右，几分钱）。
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI
from qdrant_client import models

from app.config import get_settings
from rag.chunker import chunk_pages
from rag.cleaner import clean_pages
from rag.embedder import get_embedder
from rag.parser import parse_pdf
from rag.retriever import get_client

PDF = "data/langchain_intro.pdf"
CAND_CSV = Path("eval/evalset_candidates.csv")
GOLDEN = Path("eval/golden_set.json")
QC_COLLECTION = "evalset_qc"       # 质检用的临时集合，跑完就删
CONCURRENCY = 4                     # 同时问几个块，太快容易被限流

# 出题风格：真实用户里两种都有，混着出才公平
#   loose —— 口语描述，不带材料里的术语原名 → 考语义检索
#   term  —— 直接问某个类/函数/参数名     → 考关键词检索
STYLE_LOOSE = (
    "问题里**不要**出现材料中的类名、函数名、参数名等专有术语，"
    "用你自己的话把这件事描述出来，就像用户还不知道这东西叫什么"
)
STYLE_TERM = (
    "问题里要**直接点名**材料中出现的那个具体类名/函数名/参数名，"
    "就像用户已经知道名字、只是想查它怎么用"
)

PROMPT = """你是 RAG 评测集的出题人。下面是一份英文技术文档里的一个片段。

【文档主题】{kb}
【片段（来自第 {page} 页）】
{text}

请出 1 个**中文**问题（用户是中文开发者，正在查这份英文文档）。要求：
1. 这个问题必须**只靠上面这段材料**就能回答
2. 口语化，像用户在搜索框里输入的，不要书面考试腔
3. 要问材料里**特有的具体内容**，不要"这个文档讲了什么"这种大而无当的问题
4. {style}
5. 答案要点 1~2 句，**只依据材料**，不要补充材料里没有的知识
6. 如果这段材料没有可提问的实质内容（纯代码、纯链接列表、纯目录），
   question 返回 null

正例：材料写「用 Y 把文档按分隔符切成小块，直到达到指定大小」
      → 问「用什么办法能把长文档切小一点」，答「用 Y，按分隔符切到指定大小」
反例（不要这样）：问「这个文档的主要内容包括哪些」
      → 太空泛，检索时任何块都沾边，评估不出真实能力

只输出 JSON：{{"question": "...", "answer": "..."}}"""

REFUSAL_PROMPT = """你在为一个 RAG 知识库设计"应该拒答"的测试题。

【知识库主题】{kb}
【知识库内容概览（开头几页节选）】
{head}

请生成 {total} 个中文问题，分三类：
A. 属于这个知识库主题、但上面这份文档**没有讲到**的具体问题 —— {na} 个
   这是最难判、最有价值的一类：主题相关但材料缺内容，
   正确行为是拒答，而不是拿模型自己的常识编一个答案。
   正例：文档讲怎么切分文档、怎么建向量库，但没讲怎么做混合检索
         → 问「混合检索里关键词和向量的权重怎么调」
   反例：问「今天天气怎么样」—— 这类检索层一筛就掉，评估价值低
B. 关于提问者所在组织内部事务的问题（报销、考勤、审批、排班、人事这类）—— {nb} 个
   这类无论材料里有没有，都必须拒答：内部资料不能用通用知识硬套。
   要用"我们公司""我们团队"这种口吻提问，不要指名某家具体公司
C. 与主题完全无关的通用问题 —— {nc} 个（只用来做烟囱测试）

要求：问题都要像真实用户随口问的，中文，不要解释，不要编号。
只输出 JSON 数组：[{{"kind": "A", "question": "..."}}, {{"kind": "B", ...}}]"""


def _extract_json(raw: str):
    """容错解析 LLM 输出：剥掉 ```json 围栏，再抓第一个 JSON 值。

    不能直接 json.loads——模型经常在前后加一句"好的，以下是..."。
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 兜底：抓第一个 { 或 [ 到最后一个 } 或 ]
    for left, right in (("{", "}"), ("[", "]")):
        start, end = text.find(left), text.rfind(right)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _is_useful(text: str) -> bool:
    """判断一个 chunk 值不值得出题。

    剔掉三类：太短(没有完整信息) / 链接堆积(导航页) / 基本是代码。
    代码块本身能出题，但这份 PDF 里的代码块很多是跨页重复的片段，
    出出来的题会集中在同一段代码上，把评估集带偏。
    """
    if len(text) < 150:
        return False
    if len(re.findall(r"https?://", text)) >= 2:
        return False
    # 代码行特征：以缩进/符号开头的行占比过高
    lines = [ln for ln in text.splitlines() if ln.strip()]
    codey = sum(1 for ln in lines
                if re.match(r"^\s*([{}()\[\]#>*-]|[a-zA-Z_]+\s*=|import |from |def )",
                            ln))
    return bool(lines) and codey / len(lines) < 0.5


def pick_chunks(chunks: list, n: int) -> list:
    """按页轮转抽块：每页先取一个，不够再回头取第二个。

    直接随机抽会让候选集中在内容多的页，评估集的页码分布就偏了。
    """
    by_page: dict[int, list] = {}
    seen: set[str] = set()
    for c in chunks:
        if not _is_useful(c.text):
            continue
        key = c.text[:80]          # 跨页重复块去重（这份 PDF 有重复代码块）
        if key in seen:
            continue
        seen.add(key)
        by_page.setdefault(c.page_numbers[0], []).append(c)

    picked, depth = [], 0
    while len(picked) < n:
        added = False
        for page in sorted(by_page):
            if depth < len(by_page[page]):
                picked.append(by_page[page][depth])
                added = True
                if len(picked) >= n:
                    break
        if not added:              # 块已经被抽干了
            break
        depth += 1
    return picked


def build_qc_index(chunks: list, embedder) -> None:
    """把切好的块灌进临时集合，用来给候选题做免费质检。"""
    client = get_client()
    client.recreate_collection(
        collection_name=QC_COLLECTION,
        vectors_config=models.VectorParams(size=embedder.dim,
                                           distance=models.Distance.COSINE),
    )
    vectors = embedder.embed_documents([c.text for c in chunks])
    client.upsert(collection_name=QC_COLLECTION, points=[
        models.PointStruct(
            id=c.chunk_index, vector=v,
            payload={"page_numbers": c.page_numbers},
        )
        for c, v in zip(chunks, vectors)
    ])


def qc_one(question: str, embedder) -> tuple[float, list[int], str]:
    """本地跑一次向量检索：返回 (top1 相似度, top1 页码, top1 原文前 60 字)。

    用途：人工筛题时看这几列——
      可答题却没命中标注页 → 要么标注错了，要么题太难，值得重点看
      应拒答题 top1 很高     → 属于常态（关键词重合但材料没给答案），
                              但要看"最相关材料"那列确认文档里确实没写
    """
    vec = embedder.embed_query(question)
    points = get_client().query_points(
        collection_name=QC_COLLECTION, query=vec, limit=1,
    ).points
    if not points:
        return 0.0, [], ""
    return (points[0].score,
            points[0].payload.get("page_numbers", []),
            points[0].payload.get("text", "")[:60].replace("\n", " "))


async def gen_one(client: AsyncOpenAI, sem: asyncio.Semaphore,
                  chunk, page: int, style: str, kb: str) -> dict | None:
    """给一个 chunk 出一道可答题。"""
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=get_settings().llm_model,
                messages=[{"role": "user", "content": PROMPT.format(
                    kb=kb, page=page, text=chunk.text, style=style)}],
                temperature=0.7,   # 出题要一点多样性，不然句式全一样
                max_tokens=300,
            )
            data = _extract_json(resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001 - 单块失败不该毁掉整轮
            print(f"  [WARN] 第 {page} 页出题失败: {type(exc).__name__}: {exc}")
            return None

    if not isinstance(data, dict) or not data.get("question"):
        return None
    return {
        "kind": "answerable",
        "question": str(data["question"]).strip(),
        "answer": str(data.get("answer") or "").strip(),
        "pages": [page],
        "source": chunk.text[:60].replace("\n", " "),
    }


async def gen_refusals(client: AsyncOpenAI, kb: str, head: str) -> list[dict]:
    """生成一批应拒答题（主题内缺内容 / 内部事务 / 完全无关）。"""
    resp = await client.chat.completions.create(
        model=get_settings().llm_model,
        messages=[{"role": "user", "content": REFUSAL_PROMPT.format(
            kb=kb, head=head, total=14, na=8, nb=4, nc=2)}],
        temperature=0.8,
        max_tokens=800,
    )
    data = _extract_json(resp.choices[0].message.content or "")
    if not isinstance(data, list):
        print(f"  [WARN] 应拒答题解析失败，原始输出：{resp.choices[0].message.content[:200]}")
        return []

    items = []
    for i, entry in enumerate(data):
        if isinstance(entry, str):
            entry = {"kind": "A", "question": entry}
        q = str(entry.get("question") or "").strip()
        if not q:
            continue
        kind = str(entry.get("kind") or "A").upper()
        label = {"A": "应拒-主题内缺内容", "B": "应拒-内部事务",
                 "C": "应拒-完全无关"}.get(kind, "应拒-主题内缺内容")
        items.append({"kind": label, "question": q, "answer": "",
                      "pages": None, "source": f"（LLM 生成 #{i + 1}）"})
    return items


def write_csv(rows: list[dict]) -> None:
    """写候选表。utf-8-sig 是为了 Excel 双击打开不乱码（踩过 BOM 的坑）。"""
    CAND_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = ["编号", "类型", "问题", "标注页码", "参考答案",
              "检索top1", "检索命中", "来源块摘要", "保留"]
    with CAND_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in enumerate(rows, start=1):
            writer.writerow([
                i, row["kind"], row["question"],
                "/".join(map(str, row["pages"])) if row["pages"] else "",
                row["answer"], f"{row['qc_score']:.3f}", row["qc_hit"],
                row["source"], "y",
            ])


def do_generate(n_answer: int) -> None:
    settings = get_settings()
    print(f"文档：{PDF}｜主题：{settings.kb_description}\n")

    pages = clean_pages(parse_pdf(PDF))
    chunks = chunk_pages(pages, chunk_size=300, overlap=30)
    embedder = get_embedder()
    print(f"切分完成：{len(pages)} 页 → {len(chunks)} 块")

    picked = pick_chunks(chunks, n_answer)
    print(f"抽中 {len(picked)} 块用于出题（已剔除过短/链接堆积/重复块）\n")

    # 质检集合：和正式入库同参数，这样 top1 分数才有参考价值
    build_qc_index(chunks, embedder)

    async def run() -> list[dict]:
        client = AsyncOpenAI(api_key=settings.llm_api_key,
                             base_url=settings.llm_base_url)
        sem = asyncio.Semaphore(CONCURRENCY)

        # 风格交替：偶数块走"不点名"，奇数块走"点名"，两种检索能力都覆盖到
        tasks = [
            gen_one(client, sem, c, c.page_numbers[0],
                    STYLE_TERM if i % 2 else STYLE_LOOSE,
                    settings.kb_description)
            for i, c in enumerate(picked)
        ]
        print(f"出题中（{len(tasks)} 块，并发 {CONCURRENCY}）...")
        answerable = [r for r in await asyncio.gather(*tasks) if r]

        print("生成应拒答题中...")
        head = "\n".join(p.text for p in pages[:3])[:3000]
        refusals = await gen_refusals(client, settings.kb_description, head)
        return answerable + refusals

    rows = asyncio.run(run())
    if not rows:
        print("[错误] 一道题都没生成出来，检查 .env 里的 LLM 配置")
        return

    # 免费质检：每题在本地跑一次向量检索，把分数和命中写进表里
    print("本地质检中...")
    for row in rows:
        score, hit_pages, top_text = qc_one(row["question"], embedder)
        row["qc_score"] = score
        if row["pages"] is None:
            # 应拒答题 top1 高是常态：关键词与语料重合、但材料没给答案。
            # 机器判不了"文档里到底有没有答案"，所以把最相关的那块原文
            # 一起放进表里，人工扫一眼就能定。
            row["qc_hit"] = "需人工确认"
            row["source"] = f"[最相关·第{'/'.join(map(str, hit_pages))}页] {top_text}"
        elif set(hit_pages) & set(row["pages"]):
            row["qc_hit"] = "✅命中"
        else:
            row["qc_hit"] = f"❌命中[{','.join(map(str, hit_pages))}]"

    write_csv(rows)
    get_client().delete_collection(QC_COLLECTION)

    n_ans = sum(1 for r in rows if r["pages"])
    print(f"\n已写出 {CAND_CSV}（{len(rows)} 道：{n_ans} 可答 + "
          f"{len(rows) - n_ans} 应拒答）")
    print("下一步：用 Excel 打开筛选——")
    print("  · 删掉不成立/重复/太泛的题；不想删就把「保留」列改成 n")
    print("  · 目标是 30 题上下（可答题多留一些，应拒答留 8~10 道即可）")
    print("  · 「检索命中」列是质检结果：可答题 ❌ 的、应拒答要看的「最相关」原文")
    print("  · 确认完运行：python scripts/gen_evalset.py --finalize")
    if GOLDEN.exists():
        old = json.loads(GOLDEN.read_text(encoding="utf-8"))
        print(f"\n⚠️ {GOLDEN} 已存在（{old['stats']['total']} 题），"
              f"筛完不跑 --finalize 的话评估用的还是这套旧题")


def do_finalize() -> None:
    """人工筛完的 CSV → 正式题集 JSON。"""
    if not CAND_CSV.exists():
        raise SystemExit(f"找不到 {CAND_CSV}，先跑 python scripts/gen_evalset.py")

    items, dropped = [], 0
    with CAND_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("保留", "")).strip().lower() in ("n", "no", "0", "否"):
                dropped += 1
                continue
            question = (row.get("问题") or "").strip()
            if not question:
                dropped += 1
                continue
            page_text = (row.get("标注页码") or "").strip()
            pages = ([int(p) for p in re.split(r"[/,，、\s]+", page_text) if p]
                     if page_text else None)
            kind = (row.get("类型") or "").strip()
            items.append({
                "q": question,
                "pages": pages,
                "reference": (row.get("参考答案") or "").strip() or None,
                "expect": "answerable" if pages else (
                    "general" if "完全无关" in kind else "refuse"),
                "kind": kind,
            })

    n_ans = sum(1 for i in items if i["pages"])
    golden = {
        "version": 1,
        "corpus": PDF,
        "chunk_config": {"chunk_size": 300, "overlap": 30},
        "stats": {"total": len(items), "answerable": n_ans,
                  "refusal": len(items) - n_ans},
        "items": items,
    }
    GOLDEN.write_text(json.dumps(golden, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"已写出 {GOLDEN}：{len(items)} 题"
          f"（{n_ans} 可答 + {len(items) - n_ans} 应拒答），筛掉 {dropped} 条")
    print("接下来：python scripts/eval_retrieval.py   # 会自动用新题集")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="生成多少道可答题（默认40）")
    ap.add_argument("--finalize", action="store_true",
                    help="读回人工筛过的 CSV，生成 eval/golden_set.json")
    args = ap.parse_args()

    if args.finalize:
        do_finalize()
    else:
        do_generate(args.n)


if __name__ == "__main__":
    main()
