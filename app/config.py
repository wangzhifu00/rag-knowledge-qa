"""全局配置：从 .env 读取，带类型校验。
配置从哪来，全项目的“通讯录 + 钥匙串”
用法：
    from app.config import get_settings
    settings = get_settings()      # 全局单例（lru_cache）
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", #从项目根目录 .env 读取
        env_file_encoding="utf-8",  #
        extra="ignore",  # .env 里有但没声明的变量直接忽略，不报错
    )

    # ---------- PostgreSQL 元数据库 ----------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "raguser"
    postgres_password: str = "changeme_pg_pwd"
    postgres_db: str = "ragdb"

    # ---------- Redis 缓存----------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # ---------- Qdrant 向量数据库----------
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # ---------- LLM ----------
    # 用哪家由 LLM_PROVIDER 决定（deepseek / openai / qwen / glm），改 .env 即可切换
    llm_provider: str = "deepseek"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # ---------- Embedding 模型 ----------
    # 默认 bge-small-zh-v1.5：fastembed/ONNX，512 维，无 torch 依赖，CPU 上几十毫秒/块。
    # 换成多语言大模型（如 BAAI/bge-m3，1024 维）时改这里，然后跑 scripts/reindex.py
    # 重算向量——原文在 PG 里不动，所以换模型前后的效果对比是干净的。
    # 注意：字段名故意不叫 embedding_model —— .env 里那个 EMBEDDING_MODEL
    # 是早期 OpenAI embedding 方案留下的死配置（text-embedding-3-small），
    # 沿用同名会被它悄悄覆盖掉。
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"

    # ---------- 应用 ----------
    app_port: int = 8000
    log_level: str = "INFO"

    # ---------- 设备与精度（4GB 显卡的生存配置）----------
    # auto / cuda / cpu。两个模型都 auto 时都会上 GPU——
    # 3050 Ti 只有 4GB，fp32 的两个 568M 模型必然 OOM，
    # 所以默认开半精度把它们压到各 ~1.15GB。
    # 若仍不够：把 rerank_device 设成 cpu，只让 embedding 吃显卡。
    embedding_device: str = "auto"
    rerank_device: str = "auto"
    gpu_fp16: bool = True

    # ---------- 检索质量门槛（前置拒答） ----------
    # top1 向量相似度低于该值时直接拒答、不调 LLM：
    # 既堵住"材料不够模型硬编"的幻觉，又省 token。
    # 定值依据：语料内正例 top1≈0.63，库外问题≈0.40，取中间偏保守；
    # 实测 bge-small 正负例分数重叠，此门槛区分度不足(见评估)，
    # rerank 模式下应改用 rerank_min_score。
    retrieval_min_score: float = 0.45

    # rerank 模式的拒答门槛：top1 rerank 相关度概率(0~1)低于该值拒答。
    # 0.5 是暂定值——reranker 分数区分度高，正例通常 >0.9、负例 <0.1，
    # 评估脚本跑完 10 题看分布后再校准。环境变量 RERANK_MIN_SCORE 可覆盖。
    rerank_min_score: float = 0.5

    # ---------- 冷启动预热 ----------
    # 启动时预加载 embedding + reranker 模型，首问检索从 ~25s 降到 <1s。
    # 代价：每次进程启动多等约 8 秒（GPU 加载 2.2GB reranker）。
    # 开发时用 --reload 改一次代码就重启一次，建议 .env 里设 WARMUP_ON_START=false；
    # 部署/演示场景保持 true（默认）。
    warmup_on_start: bool = True

    # ---------- 知识库主题 ----------
    # 注入 prompt，用来区分两种"答不了"：
    #   · 话题属于本知识库但材料没写 → 拒答（内部资料不能被外部常识替代）
    #   · 话题根本不属于本知识库    → 用模型通用知识回答，并标明"非知识库内容"
    # 换语料时要跟着改，否则模型会拿旧主题做判断。
    kb_description: str = (
        "LangChain 官方文档——如何用 LangChain / Deep Agents 构建 RAG 问答应用"
        "（文档加载、切分、嵌入、向量库、智能体编排、运行与评估）"
    )

    #把零散字段拼成 psycopg 要的连接串。  psycopg 是 Python 连接 PostgreSQL 数据库最主流的驱动库
    # @property 表示调用时不用加括号：settings.postgres_dsn。
    @property
    def postgres_dsn(self) -> str:
        """
        拼接格式是标准的 postgresql://用户:密码@主机:端口/库名
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def qdrant_url(self) -> str:
        #形如：http://localhost:6333
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def postgres_dsn_async(self) -> str:
        """SQLAlchemy 异步驱动(asyncpg)专用连接串。
        区别在协议头：
        postgresql:// 是同步驱动(psycopg)用的，
        postgresql+asyncpg:// 告诉 SQLAlchemy 用异步驱动。
        """
        return self.postgres_dsn.replace("postgresql://", "postgresql+asyncpg://")

    # ---------- LLM 切换：按 provider 取对应的 key/url/model ----------
    # 生成层代码只认 llm_api_key/llm_base_url/llm_model 这三个"统一出口"，
    # 以后加 qwen/glm 就是在下面字典里多两行，业务代码零改动
    @property
    def llm_api_key(self) -> str:
        return {
            "deepseek": self.deepseek_api_key,
            "openai": self.openai_api_key,
        }[self.llm_provider]

    @property
    def llm_base_url(self) -> str:
        return {
            "deepseek": self.deepseek_base_url,
            "openai": self.openai_base_url,
        }[self.llm_provider]

    @property
    def llm_model(self) -> str:
        return {
            "deepseek": self.deepseek_model,
            "openai": self.openai_model,
        }[self.llm_provider]

#@lru_cache 让函数只在第一次调用时构造 Settings()（也就只读一次 .env、做一次校验），之后永远返回同一个对象。
@lru_cache
def get_settings() -> Settings:
    """全局唯一配置实例，作为 FastAPI 依赖注入使用。"""
    return Settings()
