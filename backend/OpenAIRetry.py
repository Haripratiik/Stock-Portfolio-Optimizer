"""
OpenAI API call wrapper with 429 rate-limit retry.

Free-tier gpt-4o-mini: 3 RPM. On 429, we wait 22s and retry (up to 4 attempts).
"""

from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

T = TypeVar('T')


def _is_rate_limit(exc: BaseException) -> bool:
    """Check if exception is OpenAI rate limit (429)."""
    err_str = str(exc).lower()
    if '429' in err_str or 'rate_limit' in err_str or 'rate limit' in err_str:
        return True
    # OpenAI SDK may wrap it
    if hasattr(exc, 'status_code') and getattr(exc, 'status_code') == 429:
        return True
    if hasattr(exc, 'response') and hasattr(exc.response, 'status_code'):
        return exc.response.status_code == 429
    return False


def _parse_retry_seconds(exc: BaseException) -> int:
    """Parse 'try again in 20s' from error message. Default 22 for 3 RPM."""
    err_str = str(exc)
    m = re.search(r'try again in (\d+)\s*s', err_str, re.I)
    if m:
        return int(m.group(1)) + 2  # add buffer
    return 22  # 3 RPM ≈ 20s between requests


def with_retry(
    fn: Callable[[], T],
    max_retries: int = 4,
) -> T:
    """
    Execute fn(). On 429 rate limit, wait and retry.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if _is_rate_limit(e) and attempt < max_retries - 1:
                wait = _parse_retry_seconds(e)
                time.sleep(wait)
                continue
            raise
    raise last_exc  # type: ignore[misc]
