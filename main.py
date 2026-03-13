"""
医学教育智能体服务 - FastAPI 主入口
Medical Education AI Agent Service

兼容 OpenAI API 协议，可被 Open WebUI / LibreChat 直接调用
"""
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from config import get_settings
from middleware.auth import AuthMiddleware, RateLimitMiddleware
from routers import chat, models, health

settings = get_settings()

# ======================================================
#  日志配置
# ======================================================
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level,
    colorize=True,
)
logger.add(
    settings.log_file,
    rotation="50 MB",
    retention="30 days",
    level=settings.log_level,
    encoding="utf-8",
)


# ======================================================
#  应用生命周期
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    logger.info("=" * 60)
    logger.info("  🏥 医学教育智能体服务 启动中...")
    logger.info(f"  LLM 提供商: {settings.llm_provider}")
    logger.info(f"  API 认证: {'已启用' if settings.enable_api_key_auth else '已禁用'}")
    logger.info(f"  流式响应: {'已启用' if settings.enable_streaming else '已禁用'}")
    logger.info(f"  Function Calling: {'已启用' if settings.enable_function_calling else '已禁用'}")
    logger.info(f"  速率限制: {settings.rate_limit_per_minute} 次/分钟")
    logger.info(f"  访问文档: http://{settings.host}:{settings.port}/docs")
    logger.info("=" * 60)
    
    # 预加载 LLM 客户端（验证配置）
    try:
        from utils.llm_client import get_llm_client
        client = get_llm_client()
        logger.info(f"✅ LLM 客户端初始化成功: {settings.llm_provider}")
    except Exception as e:
        logger.warning(f"⚠️  LLM 客户端初始化警告: {e}")
        logger.warning("   请检查 API Key 配置，服务仍将启动但 LLM 调用可能失败")
    
    yield
    
    # 关闭时
    logger.info("医学教育智能体服务 已关闭")


# ======================================================
#  FastAPI 应用
# ======================================================
app = FastAPI(
    title="医学教育智能体服务 API",
    description="""
## 🏥 医学教育智能体服务

基于大语言模型的医学教育 AI 服务，**完全兼容 OpenAI API 协议**，可直接被 Open WebUI、LibreChat 等平台调用。

### 🤖 可用智能体（通过模型名称选择）

| 模型 ID | 功能 |
|---------|------|
| `med-general` | 通用医学教育助手 |
| `med-clinical` | 临床病例分析与教学 |
| `med-pharmacology` | 药理学教学 |
| `med-anatomy` | 基础医学解剖教学 |
| `med-exam` | 考试备考辅导 |
| `med-diagnosis` | 辅助诊断教学 |

### 🔗 接入 Open WebUI

在 Open WebUI 的 **设置 → 连接 → OpenAI API** 中配置：
- **API Base URL**: `http://your-server:8000/v1`
- **API Key**: 你设置的 `SERVICE_API_KEY`

### 🔗 接入 LibreChat

在 `librechat.yaml` 中添加：
```yaml
endpoints:
  custom:
    - name: "医学教育智能体"
      apiKey: "your-service-api-key"
      baseURL: "http://your-server:8000/v1"
      models:
        default: ["med-general", "med-clinical", "med-pharmacology"]
```

### ⚠️ 免责声明
本服务仅供医学教育使用，不构成具体患者的诊疗建议。
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ======================================================
#  中间件（顺序：CORS → 速率限制 → 认证）
# ======================================================

# CORS - 允许 Open WebUI / LibreChat 跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 速率限制
if settings.enable_rate_limiting and settings.rate_limit_per_minute > 0:
    app.add_middleware(RateLimitMiddleware, calls_per_minute=settings.rate_limit_per_minute)

# API Key 认证
if settings.enable_api_key_auth:
    app.add_middleware(AuthMiddleware)


# ======================================================
#  路由注册
# ======================================================
app.include_router(health.router)
app.include_router(models.router)
app.include_router(chat.router)


# ======================================================
#  全局异常处理
# ======================================================
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": {"message": f"路径 {request.url.path} 不存在", "type": "not_found"}}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.exception(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "内部服务器错误", "type": "server_error"}}
    )


# ======================================================
#  入口
# ======================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
