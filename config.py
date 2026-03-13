"""
配置管理模块
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # LLM 后端
    llm_provider: str = Field(default="anthropic", env="LLM_PROVIDER")

    # Anthropic
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", env="ANTHROPIC_MODEL")

    # OpenAI / 兼容接口
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")

    # 服务配置
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    debug: bool = Field(default=False, env="DEBUG")

    # 安全
    service_api_key: str = Field(default="med-edu-secret-key", env="SERVICE_API_KEY")
    enable_api_key_auth: bool = Field(default=True, env="ENABLE_API_KEY_AUTH")
    secret_key: str = Field(default="change-me", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=10080, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # 智能体
    default_agent_mode: str = Field(default="general", env="DEFAULT_AGENT_MODE")
    max_context_tokens: int = Field(default=100000, env="MAX_CONTEXT_TOKENS")
    max_output_tokens: int = Field(default=4096, env="MAX_OUTPUT_TOKENS")

    # 功能开关
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    enable_function_calling: bool = Field(default=True, env="ENABLE_FUNCTION_CALLING")
    enable_usage_logging: bool = Field(default=True, env="ENABLE_USAGE_LOGGING")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")

    # 日志
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/medical_edu_agent.log", env="LOG_FILE")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # 忽略 .env 中未声明的字段，防止 ValidationError


@lru_cache()
def get_settings() -> Settings:
    return Settings()
