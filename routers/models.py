"""
模型列表路由 - GET /v1/models
兼容 OpenAI API 规范，供 Open WebUI / LibreChat 获取可用模型列表
"""
import time
from fastapi import APIRouter, Depends
from loguru import logger

from utils.schemas import ModelCard, ModelList
from prompts.system_prompts import AGENT_DESCRIPTIONS

router = APIRouter()

# 医学教育智能体模型列表
MEDICAL_AGENT_MODELS = [
    {
        "id": "med-general",
        "description": AGENT_DESCRIPTIONS["general"],
        "context_length": 200000,
    },
    {
        "id": "med-clinical",
        "description": AGENT_DESCRIPTIONS["clinical"],
        "context_length": 200000,
    },
    {
        "id": "med-pharmacology",
        "description": AGENT_DESCRIPTIONS["pharmacology"],
        "context_length": 200000,
    },
    {
        "id": "med-anatomy",
        "description": AGENT_DESCRIPTIONS["anatomy"],
        "context_length": 200000,
    },
    {
        "id": "med-exam",
        "description": AGENT_DESCRIPTIONS["exam"],
        "context_length": 200000,
    },
    {
        "id": "med-diagnosis",
        "description": AGENT_DESCRIPTIONS["diagnosis"],
        "context_length": 200000,
    },
]


@router.get("/v1/models", response_model=ModelList, tags=["Models"])
async def list_models():
    """
    获取可用的医学教育智能体模型列表。
    
    兼容 OpenAI /v1/models 接口，Open WebUI 和 LibreChat 会调用此接口获取模型选项。
    """
    created = int(time.time())
    models = [
        ModelCard(
            id=m["id"],
            created=created,
            owned_by="medical-edu-agent",
        )
        for m in MEDICAL_AGENT_MODELS
    ]
    logger.info(f"返回 {len(models)} 个医学教育智能体模型")
    return ModelList(data=models)


@router.get("/v1/models/{model_id}", tags=["Models"])
async def get_model(model_id: str):
    """获取指定模型信息"""
    for m in MEDICAL_AGENT_MODELS:
        if m["id"] == model_id:
            return ModelCard(id=m["id"], owned_by="medical-edu-agent")
    
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")
