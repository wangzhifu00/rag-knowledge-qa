"""FastAPI 应用入口。
应用的总装线
本地启动：
    uvicorn app.main:app --reload --port 8000
访问：
    http://127.0.0.1:8000/docs      交互文档
    http://127.0.0.1:8000/health    依赖组件健康检查
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from app.api.routes import chat, documents, health
from app.config import get_settings

settings = get_settings()  #全局唯一配置

#用绝对路径
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIR = BASE_DIR.parent / "frontend"   # 项目根/frontend，网页聊天界面


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", #时间 | 级别 | 模块名 | 消息
)
logger = logging.getLogger(__name__)


def _warmup_models() -> None:
    """把 embedding 和 reranker 实例化一遍，触发模型加载并常驻内存/显存。

    两个模块内部都有进程级单例（get_embedder / get_reranker），
    这里实例化后，之后的请求直接复用，不再付加载时间。
    Reranker 的模型在 __init__ 里就加载（CrossEncoder），所以拿到实例即完成预热。

    顺带把 BM25 索引也建好：它要读 PG 全量 chunk 再分词，105 块约 0.5 秒，
    放在首问会让人以为"服务卡了"。PG 没起来时这一步失败不影响服务启动。
    """
    from rag.embedder import get_embedder
    from rag.reranker import get_reranker

    get_embedder().embed_query("warmup")   # 触发 embedding 模型加载
    get_reranker()                         # 触发 reranker 模型加载

    try:
        from rag.hybrid import warmup_jieba, warmup_bm25
        warmup_jieba()
        n = warmup_bm25()
        logger.info("BM25 索引预热完成（%d 块）", n)
    except Exception:
        logger.warning("BM25 预热失败（PG 未就绪？），首个混合检索请求会现场建索引")

#FastAPI 新一代的生命周期管理方式
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("应用启动，Qdrant=%s", settings.qdrant_url)
    # 冷启动预热：把 embedding + reranker 模型提前加载进显存。
    # 不预热的话，第一个用户的第一个问题要替所有人付 ~25s 的模型加载费；
    # 预热后这笔开销挪到服务启动阶段，首问检索 <1s。
    # 开发时(--reload 改一次代码就重启一次)可在 .env 设 WARMUP_ON_START=false 关掉。
    if settings.warmup_on_start:
        import asyncio
        import time
        logger.info("开始预热模型（首次约 8~25 秒，视 GPU/CPU 而定）...")
        t0 = time.time()
        try:
            # to_thread: 模型加载是阻塞的 CPU/磁盘活，不能卡事件循环
            await asyncio.to_thread(_warmup_models)
            logger.info("模型预热完成，耗时 %.1f 秒", time.time() - t0)
        except Exception:
            logger.exception("模型预热失败（服务继续启动，首个请求会现场加载）")
    yield
    #第③步新增：关闭时释放数据库连接池，否则 uvicorn 退出会有残留连接警告
    from app.db.engine import engine

    await engine.dispose()
    logger.info("应用关闭（连接池已释放）")
    #yield 之前 = 启动时执行（以后在这里创建数据库连接池），
    #yield 之后 = 关闭时执行（释放连接池）。这就是服务"开机/关机"的钩子。


app = FastAPI(
    title="RAG Knowledge QA",
    version="0.1.0",
    description="基于 Qdrant + BGE-M3 + DeepSeek 的知识库智能问答系统",
    lifespan=lifespan,
    docs_url=None,   # 关掉默认 /docs（它从国外 CDN 拉资源，国内会白屏）
    redoc_url=None,
)


#中间件 = 请求进来先过它、响应出去也过它。
#CORS 是跨域，以后前端页面调你接口必须开着，否则浏览器拦。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 本地自托管 Swagger UI 静态资源
# mount 挂静态文件（Swagger 的 JS/CSS）；include_router 把另一个文件里的路由"装"进主应用 —— 这样业务代码可以按模块拆文件，不用全堆在 main.py。
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 网页聊天界面。注意挂的是 /ui 而不是 "/"：
# StaticFiles 挂到根路径会变成一个"什么都匹配"的兜底路由，
# 排在它后面的 /health、/chat 全被它吃掉返回 404 或 index.html。
# html=True 让 /ui/ 自动返回 index.html，静态文件里的相对路径也按 /ui/ 解析。
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    """Swagger UI，改用本地文件，内网/离线也能打开。"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon.png",
        swagger_ui_parameters={
            "docExpansion": "list",   #打开 /docs 时所有接口按列表展开
            "defaultModelsExpandDepth": 1,  #底部 "Schemas" 区域的模型展开一层
            "persistAuthorization": True,# 刷新页面后保留 Authorization 头
        },
    )

#ReDoc 是另一种文档 UI
@app.get("/redoc", include_in_schema=False)
async def custom_redoc():
    """ReDoc 同样走本地资源。"""
    from fastapi.openapi.docs import get_redoc_html
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        with_google_fonts=False,  # 关掉 Google Fonts，国内会卡
    )


@app.get("/", tags=["meta"], summary="服务信息")
async def root() -> dict:
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "ui": "/ui/",   # 网页聊天界面
    }
