"""
Rate limiting configuration for ChefPath API.

Uses slowapi to protect AI endpoints from abuse.
Configured to work with Railway's proxy using X-Forwarded-For header.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request


def get_real_ip(request: Request) -> str:
    """
    Extract the real client IP from request headers.

    Priority:
    1. X-Forwarded-For (first IP in chain) - for Railway/proxy deployments
    2. X-Real-IP - alternative proxy header
    3. request.client.host - direct connection fallback

    Args:
        request: FastAPI Request object

    Returns:
        Client IP address as string
    """
    # Railway and most proxies use X-Forwarded-For
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can be "client, proxy1, proxy2"
        # Take the first (leftmost) IP which is the original client
        return forwarded_for.split(",")[0].strip()

    # Fallback to X-Real-IP
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Final fallback to direct connection
    return get_remote_address(request)


# Initialize the rate limiter with custom key function
limiter = Limiter(key_func=get_real_ip)
