"""设备与精度：一张 4GB 显卡怎么同时放下两个 568M 的模型。

背景（踩过的坑）：
  BGE-M3(fp32, 2.1GB) + bge-reranker-v2-m3(fp32, 2.3GB) 在 RTX 3050 Ti
  Laptop(4GB) 上必然 OOM。注意报错位置——不是 predict 时 batch 太大，
  而是 __init__ 里 .to(device) 搬权重那一刻就炸了，说明模型本身装不下，
  调 batch_size 完全没用（踩过一次，白改）。

两条出路：
  1. 半精度(fp16)：两个模型各约 1.15GB，合计 2.3GB，4GB 卡够用。
     bge 系列 fp16 对检索/排序结果几乎无影响（分数小数点后第三位才动），
     代价是显存减半——这笔账在任何消费级显卡上都划算。
  2. 一个上 GPU、一个留 CPU：用 EMBEDDING_DEVICE / RERANK_DEVICE 指定。
     比如 RERANK_DEVICE=cpu，只让 embedding 吃显卡。
     注意不能反过来指望"都自动"——auto 的意思就是"有卡就上卡"，
     两个都要上卡时仍然会 OOM，必须靠精度或显式降级来解。

缓存策略：fp16 只在 GPU 上开。CPU 上 half 反而更慢（老指令集没有
fp16 加速），而且部分算子直接不支持。
"""


def resolve_device(pref: str | None) -> str:
    """把配置里的 auto/cuda/cpu 解析成实际设备。"""
    if pref and pref.strip().lower() not in ("", "auto"):
        return pref.strip().lower()
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def model_kwargs(device: str, fp16: bool) -> dict:
    """给 SentenceTransformer / CrossEncoder 的 model_kwargs。

    作用是在 from_pretrained 阶段就直接按 fp16 加载权重，
    而不是"先 fp32 上卡、再 half"——后者会多要一份临时显存，
    正是 4GB 卡上最容易炸的那一步（报错里那句 allocate 978MB 就是它）。
    """
    if device != "cuda" or not fp16:
        return {}
    import torch
    return {"torch_dtype": torch.float16}
