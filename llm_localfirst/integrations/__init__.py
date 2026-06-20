"""Optional framework integrations for llm-localfirst.

Each integration imports its provider/framework SDK lazily, so importing this
package never requires the optional extras to be installed.
"""

from __future__ import annotations

from .mcp import build_mcp_server
from .pydantic_ai import attach_worker

__all__ = ["attach_worker", "build_mcp_server"]
