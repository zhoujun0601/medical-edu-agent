"""
医学教育智能体核心编排器
负责选择智能体、注入系统提示、处理工具调用
"""
import json
from typing import List, Optional, AsyncGenerator, Tuple
from loguru import logger

from config import get_settings
from utils.schemas import (
    ChatMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatCompletionStreamResponse,
)
from utils.llm_client import get_llm_client
from prompts.system_prompts import AGENT_PROMPTS, GENERAL_SYSTEM_PROMPT
from tools.medical_tools import MEDICAL_TOOLS, MedicalToolExecutor

settings = get_settings()


def _detect_agent_mode(messages: List[ChatMessage], requested_mode: Optional[str]) -> str:
    """自动检测或使用指定的智能体模式"""
    if requested_mode and requested_mode in AGENT_PROMPTS:
        return requested_mode
    
    # 从模型名称中检测（Open WebUI 通过模型名传递模式）
    # 例如模型名为 "med-clinical" 则使用 clinical 模式
    
    # 关键词自动检测
    user_text = ""
    for msg in messages[-3:]:  # 只看最近3条消息
        if msg.role == "user" and msg.content:
            if isinstance(msg.content, str):
                user_text += msg.content.lower()
            
    keywords = {
        "clinical": ["病例", "案例", "患者", "诊断", "症状", "体征", "查房", "case", "patient"],
        "pharmacology": ["药物", "用药", "剂量", "禁忌", "药理", "drug", "medication", "dose"],
        "anatomy": ["解剖", "结构", "位置", "组织", "生理", "anatomy", "physiology"],
        "exam": ["考试", "题目", "真题", "模拟题", "执业医师", "规培", "exam", "question"],
        "diagnosis": ["诊断", "鉴别", "检查", "化验", "影像", "diagnosis", "lab", "imaging"],
    }
    
    for mode, kws in keywords.items():
        if any(kw in user_text for kw in kws):
            return mode
    
    return settings.default_agent_mode or "general"


def _inject_system_prompt(messages: List[ChatMessage], agent_mode: str) -> List[ChatMessage]:
    """注入医学教育系统提示词"""
    system_prompt = AGENT_PROMPTS.get(agent_mode, GENERAL_SYSTEM_PROMPT)
    
    # 检查是否已有系统提示
    if messages and messages[0].role == "system":
        existing = messages[0].content or ""
        if isinstance(existing, str):
            # 在现有系统提示前追加医学教育提示
            new_system = f"{system_prompt}\n\n---\n\n**用户自定义指令：**\n{existing}"
        else:
            new_system = system_prompt
        return [ChatMessage(role="system", content=new_system)] + messages[1:]
    else:
        return [ChatMessage(role="system", content=system_prompt)] + messages


def _get_model_agent_mode(model_name: str) -> Optional[str]:
    """从模型名称解析智能体模式"""
    model_to_mode = {
        "med-general": "general",
        "med-clinical": "clinical",
        "med-pharmacology": "pharmacology",
        "med-anatomy": "anatomy",
        "med-exam": "exam",
        "med-diagnosis": "diagnosis",
    }
    return model_to_mode.get(model_name)


class MedicalAgentOrchestrator:
    """医学教育智能体编排器"""

    def __init__(self):
        self.llm = get_llm_client()

    async def process(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """非流式处理"""
        messages, agent_mode, max_tokens = self._prepare_request(request)
        
        # 工具调用循环（最多3轮）
        for iteration in range(3):
            tools = MEDICAL_TOOLS if settings.enable_function_calling else None
            
            response = await self.llm.chat(
                messages=messages,
                model=request.model,
                temperature=request.temperature or 0.7,
                max_tokens=max_tokens,
                tools=tools,
            )
            
            choice = response.choices[0]
            
            # 检查是否有工具调用
            if (choice.finish_reason == "tool_calls" and 
                choice.message.tool_calls and
                settings.enable_function_calling):
                
                messages = await self._handle_tool_calls(messages, choice.message)
                continue
            
            # 覆盖模型名称为请求中的模型名
            response.model = request.model
            logger.info(f"[{agent_mode}] 完成响应, tokens={response.usage.total_tokens}")
            return response
        
        return response

    async def stream_process(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        """流式处理"""
        messages, agent_mode, max_tokens = self._prepare_request(request)
        
        async for chunk in self.llm.stream_chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature or 0.7,
            max_tokens=max_tokens,
        ):
            chunk.model = request.model
            yield chunk

    def _prepare_request(
        self, request: ChatCompletionRequest
    ) -> Tuple[List[ChatMessage], str, int]:
        """准备请求：注入提示词，确定智能体模式"""
        # 确定智能体模式
        agent_mode = (
            _get_model_agent_mode(request.model) or
            request.agent_mode or
            _detect_agent_mode(request.messages, None)
        )
        
        # 注入系统提示
        messages = _inject_system_prompt(list(request.messages), agent_mode)
        
        # 确定最大输出 token
        max_tokens = min(
            request.max_tokens or settings.max_output_tokens,
            settings.max_output_tokens
        )
        
        logger.info(f"智能体模式: {agent_mode}, 消息数: {len(messages)}, max_tokens: {max_tokens}")
        return messages, agent_mode, max_tokens

    async def _handle_tool_calls(
        self, messages: List[ChatMessage], assistant_msg
    ) -> List[ChatMessage]:
        """处理工具调用，返回更新后的消息列表"""
        # 添加 assistant 消息（含工具调用）
        messages.append(ChatMessage(
            role="assistant",
            content=assistant_msg.content,
            tool_calls=[tc.model_dump() for tc in (assistant_msg.tool_calls or [])],
        ))
        
        # 执行每个工具并添加结果
        for tool_call in (assistant_msg.tool_calls or []):
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            
            result = MedicalToolExecutor.execute(tool_call.function.name, args)
            logger.info(f"工具调用: {tool_call.function.name} -> {result[:100]}...")
            
            messages.append(ChatMessage(
                role="tool",
                tool_call_id=tool_call.id,
                content=result,
            ))
        
        return messages
