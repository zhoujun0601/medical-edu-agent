"""
Chat Completions 路由 - POST /v1/chat/completions
核心 API 端点，兼容 OpenAI Chat Completions 规范
"""
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from config import get_settings
from utils.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStreamResponse, ErrorResponse, ErrorDetail
)
from agents.orchestrator import MedicalAgentOrchestrator

router = APIRouter()
settings = get_settings()

# 全局 orchestrator 实例
_orchestrator: MedicalAgentOrchestrator | None = None

def get_orchestrator() -> MedicalAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MedicalAgentOrchestrator()
    return _orchestrator


async def _stream_generator(
    request: ChatCompletionRequest,
    orchestrator: MedicalAgentOrchestrator
) -> AsyncGenerator[str, None]:
    """生成 SSE 流式数据"""
    try:
        async for chunk in orchestrator.stream_process(request):
            data = chunk.model_dump(exclude_none=True)
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"流式生成错误: {e}")
        error_data = {
            "error": {
                "message": str(e),
                "type": "internal_error",
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@router.post(
    "/v1/chat/completions",
    tags=["Chat"],
    summary="Chat Completions（OpenAI 兼容）",
    response_model=ChatCompletionResponse,
)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """
    OpenAI 兼容的 Chat Completions 接口。
    
    - 支持流式（stream=true）和非流式响应
    - 支持 6 种医学教育智能体模式（通过模型名称区分）
    - 支持 Function Calling / Tool Use
    
    **可用模型：**
    - `med-general` - 通用医学教育助手
    - `med-clinical` - 临床病例分析
    - `med-pharmacology` - 药理学教学
    - `med-anatomy` - 基础医学解剖
    - `med-exam` - 考试备考辅导
    - `med-diagnosis` - 辅助诊断教学
    
    **示例（curl）：**
    ```bash
    curl -X POST http://localhost:8000/v1/chat/completions \\
      -H "Authorization: Bearer your-api-key" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "med-clinical",
        "messages": [{"role": "user", "content": "请分析一个急性心肌梗死的典型病例"}],
        "stream": true
      }'
    ```
    """
    client_ip = raw_request.client.host if raw_request.client else "unknown"
    
    try:
        logger.info(
            f"收到请求: model={request.model}, "
            f"messages={len(request.messages)}, "
            f"stream={request.stream}, "
            f"client={client_ip}"
        )
        
        orchestrator = get_orchestrator()
        
        # 验证消息列表非空
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "messages 不能为空", "type": "invalid_request_error"}}
            )
        
        # 流式响应
        if request.stream and settings.enable_streaming:
            return StreamingResponse(
                _stream_generator(request, orchestrator),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
                }
            )
        
        # 非流式响应
        response = await orchestrator.process(request)
        return response

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"请求参数错误: {e}")
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": str(e), "type": "invalid_request_error"}}
        )
    except Exception as e:
        logger.exception(f"处理请求时发生错误: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"内部服务器错误: {str(e)}", "type": "server_error"}}
        )
