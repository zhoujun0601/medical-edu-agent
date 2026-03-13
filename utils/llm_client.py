"""
LLM 后端客户端 - 统一封装 Anthropic 和 OpenAI 接口
"""
import time
import uuid
import json
from typing import AsyncGenerator, List, Optional, Any, Dict
from loguru import logger

from config import get_settings
from utils.schemas import (
    ChatMessage, UsageInfo, ChatCompletionMessage,
    ChatCompletionChoice, ChatCompletionResponse,
    ChatCompletionStreamResponse, ChatCompletionStreamChoice,
    DeltaMessage, ToolCall, FunctionCall
)

settings = get_settings()


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数量（按字符数 / 3.5 估算英文，中文 / 1.5）"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 3.5)


def _extract_text_content(content: Any) -> str:
    """统一提取消息内容为字符串"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else 
            (part.text if hasattr(part, "text") else "")
            for part in content
        )
    return str(content) if content else ""


# ======================================================
#  Anthropic 客户端
# ======================================================
class AnthropicClient:
    def __init__(self):
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            self.model = settings.anthropic_model
        except ImportError:
            raise RuntimeError("请安装 anthropic 库: pip install anthropic")

    def _convert_messages(self, messages: List[ChatMessage]):
        """将 OpenAI 格式消息转换为 Anthropic 格式"""
        system_prompt = None
        converted = []
        
        for msg in messages:
            content = _extract_text_content(msg.content)
            if msg.role == "system":
                system_prompt = content
            elif msg.role in ("user", "assistant"):
                converted.append({"role": msg.role, "content": content})
            elif msg.role == "tool":
                # 工具结果转为 user 消息
                converted.append({"role": "user", "content": f"[Tool Result]: {content}"})
        
        return system_prompt, converted

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> ChatCompletionResponse:
        system_prompt, converted = self._convert_messages(messages)
        
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": min(temperature, 1.0),
            "messages": converted,
        }
        if system_prompt:
            params["system"] = system_prompt

        response = await self.client.messages.create(**params)
        
        content = response.content[0].text if response.content else ""
        finish_reason = "stop" if response.stop_reason == "end_turn" else response.stop_reason
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            model=model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
                finish_reason=finish_reason,
            )],
            usage=UsageInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        system_prompt, converted = self._convert_messages(messages)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": min(temperature, 1.0),
            "messages": converted,
        }
        if system_prompt:
            params["system"] = system_prompt

        # 发送角色 delta
        yield ChatCompletionStreamResponse(
            id=completion_id, created=created, model=model,
            choices=[ChatCompletionStreamChoice(
                index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
            )]
        )

        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield ChatCompletionStreamResponse(
                    id=completion_id, created=created, model=model,
                    choices=[ChatCompletionStreamChoice(
                        index=0, delta=DeltaMessage(content=text), finish_reason=None
                    )]
                )
        
        # 结束标记
        yield ChatCompletionStreamResponse(
            id=completion_id, created=created, model=model,
            choices=[ChatCompletionStreamChoice(
                index=0, delta=DeltaMessage(), finish_reason="stop"
            )]
        )


# ======================================================
#  OpenAI / 兼容客户端
# ======================================================
class OpenAIClient:
    def __init__(self):
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            self.model = settings.openai_model
        except ImportError:
            raise RuntimeError("请安装 openai 库: pip install openai")

    def _convert_messages(self, messages: List[ChatMessage]) -> List[Dict]:
        result = []
        for msg in messages:
            content = _extract_text_content(msg.content)
            result.append({"role": msg.role, "content": content})
        return result

    async def chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        converted = self._convert_messages(messages)
        
        params = dict(
            model=self.model,
            messages=converted,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            params["tools"] = tools

        response = await self.client.chat.completions.create(**params)
        choice = response.choices[0]
        msg = choice.message
        
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=FunctionCall(name=tc.function.name, arguments=tc.function.arguments)
                ) for tc in msg.tool_calls
            ]

        return ChatCompletionResponse(
            id=response.id,
            model=model,
            choices=[ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=msg.content,
                    tool_calls=tool_calls,
                ),
                finish_reason=choice.finish_reason,
            )],
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        )

    async def stream_chat(
        self,
        messages: List[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        converted = self._convert_messages(messages)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=converted,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            
            yield ChatCompletionStreamResponse(
                id=chunk.id or completion_id,
                created=chunk.created or created,
                model=model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role=delta.role,
                        content=delta.content,
                    ),
                    finish_reason=choice.finish_reason,
                )]
            )


# ======================================================
#  统一工厂函数
# ======================================================
_client_cache: Dict[str, Any] = {}

def get_llm_client():
    """获取 LLM 客户端（缓存单例）"""
    provider = settings.llm_provider
    if provider not in _client_cache:
        if provider == "anthropic":
            _client_cache[provider] = AnthropicClient()
        elif provider in ("openai", "openai_compatible"):
            _client_cache[provider] = OpenAIClient()
        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")
        logger.info(f"已初始化 LLM 客户端: {provider}")
    return _client_cache[provider]
