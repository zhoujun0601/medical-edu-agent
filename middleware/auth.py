"""
认证中间件 - API Key 验证
"""
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from loguru import logger

from config import get_settings

settings = get_settings()
security = HTTPBearer(auto_error=False)

# 跳过认证的路径
SKIP_AUTH_PATHS = {"/", "/health", "/v1/health", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key 认证中间件"""

    async def dispatch(self, request: Request, call_next):
        if not settings.enable_api_key_auth:
            return await call_next(request)

        # 跳过特定路径
        if request.url.path in SKIP_AUTH_PATHS:
            return await call_next(request)

        # 提取 API Key
        api_key = self._extract_api_key(request)

        if not api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "缺少 API Key。请在 Authorization 头中提供 Bearer <api_key>",
                        "type": "authentication_error",
                        "code": "missing_api_key"
                    }
                }
            )

        if not self._verify_api_key(api_key):
            logger.warning(f"无效的 API Key 尝试: {api_key[:8]}... from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "无效的 API Key",
                        "type": "authentication_error",
                        "code": "invalid_api_key"
                    }
                }
            )

        return await call_next(request)

    def _extract_api_key(self, request: Request) -> str | None:
        """从请求中提取 API Key"""
        # 1. Authorization: Bearer <key>
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        
        # 2. x-api-key 头（某些客户端使用）
        x_api_key = request.headers.get("x-api-key", "")
        if x_api_key:
            return x_api_key.strip()
        
        # 3. 查询参数 api_key（不推荐，但兼容性考虑）
        api_key = request.query_params.get("api_key", "")
        if api_key:
            return api_key.strip()
        
        return None

    def _verify_api_key(self, api_key: str) -> bool:
        """验证 API Key"""
        return api_key == settings.service_api_key


class RateLimitMiddleware(BaseHTTPMiddleware):
    """简单的内存速率限制中间件"""
    
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self._requests: dict = {}  # {client_ip: [timestamps]}

    async def dispatch(self, request: Request, call_next):
        if not settings.enable_rate_limiting or self.calls_per_minute == 0:
            return await call_next(request)

        if request.url.path in SKIP_AUTH_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        
        import time
        now = time.time()
        window = 60  # 1分钟窗口
        
        # 清理旧记录
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if now - t < window
            ]
        else:
            self._requests[client_ip] = []
        
        # 检查速率
        if len(self._requests[client_ip]) >= self.calls_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": f"请求频率超限，每分钟最多 {self.calls_per_minute} 次请求",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                },
                headers={"Retry-After": "60"}
            )
        
        self._requests[client_ip].append(now)
        return await call_next(request)
