"""
健康检查和服务信息路由
"""
import time
from fastapi import APIRouter
from loguru import logger

from config import get_settings
from utils.schemas import HealthResponse
from prompts.system_prompts import AGENT_DESCRIPTIONS

router = APIRouter()
settings = get_settings()

_start_time = time.time()


@router.get("/", tags=["Info"], summary="服务根路径")
async def root():
    return {
        "service": "医学教育智能体服务",
        "version": "1.0.0",
        "description": "基于大语言模型的医学教育 AI 服务，兼容 OpenAI API 协议",
        "docs": "/docs",
        "health": "/health",
        "models": "/v1/models",
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
@router.get("/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """服务健康检查"""
    return HealthResponse(
        status="ok",
        service="medical-edu-agent",
        version="1.0.0",
        agents=list(AGENT_DESCRIPTIONS.keys()),
        llm_provider=settings.llm_provider,
    )


@router.get("/v1/agents", tags=["Info"], summary="获取所有智能体信息")
async def list_agents():
    """获取所有可用智能体的详细描述"""
    agents = []
    for agent_id, description in AGENT_DESCRIPTIONS.items():
        agents.append({
            "id": agent_id,
            "model_id": f"med-{agent_id}",
            "description": description,
        })
    return {"agents": agents, "total": len(agents)}


@router.get("/v1/stats", tags=["Info"], summary="服务统计信息")
async def service_stats():
    """服务运行状态统计"""
    uptime = time.time() - _start_time
    hours, remainder = divmod(int(uptime), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return {
        "status": "running",
        "uptime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        "uptime_seconds": int(uptime),
        "llm_provider": settings.llm_provider,
        "features": {
            "streaming": settings.enable_streaming,
            "function_calling": settings.enable_function_calling,
            "api_key_auth": settings.enable_api_key_auth,
            "rate_limiting": settings.enable_rate_limiting,
        }
    }
