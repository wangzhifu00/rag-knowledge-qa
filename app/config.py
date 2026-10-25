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

    # ---------- 应用 ----------
    app_port: int = 8000
    log_level: str = "INFO"

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
