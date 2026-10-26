"""拒答边界回归：换 LLM 之后，那条"能答 / 该拒"的线有没有移动。

为什么需要它：
  检索层的评估（eval_retrieval）量的是"找没找对块"，答案层的 RAGAS 量的是
  "答得好不好"。但两者都测不到一件事——**换生成模型之后，同一个 prompt
  还会不会做出同样的拒答判断**。实测过：qwen-max 把「这个教程里索引是怎么更新的」
  误拒了（材料里明明写了"按文档变化按计划刷新"），而检索和 RAGAS 都不会报这种错。
  拒答是这套系统的卖点，也是面试时最容易被追问的地方，换模型必须单独验。

判据（与 app/api/routes/chat.py 完全一致，不另立标准）：
  答 <  以"根据现有资料无法回答"开头
  通用 < 以"（以下不属于知识库内容）"开头
  答了 < 其余

用法（项目根目录）：
    python scripts/eval_boundary.py              # 全部 29 题
    python scripts/eval_boundary.py --limit 5    # 先抽 5 题试水（省 token）
    python scripts/eval_boundary.py --show-answers   # 连回答正文一起打印

成本：每题一次生成调用（约 1.7k 输入 + 200 输出 token）。29 题 ≈ 5 万 token。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from rag.embedder import get_embedder
from rag.generator import generate_answer
from rag.hybrid import hybrid_search, warmup_jieba

GOLDEN = Path("eval/golden_set.json")
REFUSAL_PREFIX = "根据现有资料无法回答"
GENERAL_PREFIX = "（以下不属于知识库内容）"

# 期望 → 可接受的判决。拒答题判成"通用回答"算错（说明它没意识到这是库内话题），
# 通用题判成"拒答"也算错（按产品定位这类该用模型自身知识答）。
EXPECT_TO_VERDICT = {
    "answerable": "answered",
    "refuse": "refused",
    "general": "general",
}

VERDICT_CN = {"answered": "作答", "refused": "拒答", "general": "通用答"}


def classify(answer: str) -> str:
    """把回答归到三类之一。和 API 里的判定口径逐字对齐。"""
    head = (answer or "").lstrip()
    if head.startswith(REFUSAL_PREFIX):
        return "refused"
    if head.startswith(GENERAL_PREFIX):
        return "general"
    return "answered"


def load_items(limit: int | None) -> list[dict]:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    items = data["items"]
    return items[:limit] if limit else items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（省 token）")
    ap.add_argument("--show-answers", action="store_true", help="打印回答正文")
    args = ap.parse_args()

    settings = get_settings()
    items = load_items(args.limit)

    print(f"题集：eval/golden_set.json（{len(items)} 题）")
    print(f"生成模型：{settings.llm_model}")
    print("判据：以「根据现有资料无法回答」开头=拒答；以「（以下不属于知识库内容）」开头=通用答\n")

    embedder = get_embedder()
    warmup_jieba()

    rows, wrong = [], []
    for i, it in enumerate(items, start=1):
        q = it["q"]
        expect = it.get("expect", "answerable")
        hits = hybrid_search(q, embedder, top_k=5, rerank=True)
        answer = generate_answer(q, hits)
        verdict = classify(answer)

        want = EXPECT_TO_VERDICT[expect]
        ok = verdict == want
        rows.append((q, expect, verdict, ok))
        if not ok:
            wrong.append((q, expect, verdict, answer))

        mark = "✅" if ok else "❌"
        print(f"{mark} [{i:>2}] 期望{VERDICT_CN[want]} / 实际{VERDICT_CN[verdict]}  {q}")
        if args.show_answers or not ok:
            print(f"      {answer[:150].replace(chr(10), ' ')}")

    # ---- 汇总 ----
    total = len(rows)
    right = sum(1 for r in rows if r[3])
    print("\n" + "=" * 71)
    print(f"边界一致率：{right}/{total} = {right / total:.0%}")
    for expect in ("answerable", "refuse", "general"):
        sub = [r for r in rows if r[1] == expect]
        if not sub:
            continue
        good = sum(1 for r in sub if r[3])
        label = {"answerable": "该答的答了", "refuse": "该拒的拒了",
                 "general": "通用题标对了"}[expect]
        print(f"  {label:<10} {good}/{len(sub)} = {good / len(sub):.0%}")
    print("=" * 71)

    if wrong:
        print("\n判错的题（先看这几道，判断是模型变严了还是标注有问题）：")
        for q, expect, verdict, answer in wrong:
            print(f"  · [{VERDICT_CN[EXPECT_TO_VERDICT[expect]]}] → [{VERDICT_CN[verdict]}] {q}")
            print(f"    {answer[:100].replace(chr(10), ' ')}")
    else:
        print("\n全部判对：换模型没有移动拒答边界。")

    print("\n提示：误拒（该答的拒了）比漏拒更伤体验；漏拒（该拒的答了）"
          "会编内容，更伤可信度。两类要分开看，别只看总一致率。")


if __name__ == "__main__":
    main()
