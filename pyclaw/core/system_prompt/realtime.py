from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from .base import BaseLayer
from .models import LayerContext


class RealtimeLayer(BaseLayer):
    """
    Realtime layer contains data that changes every turn:
    - Current query context
    - Recent observations (if needed to be highlighted)
    """

    async def render(self, context: LayerContext) -> str:
        parts = ["<runtime_context>"]
        current_time = str(context.extra.get("current_time") or datetime.now(timezone.utc).isoformat())
        parts.append(f"current_time: {current_time}")
        if context.extra.get("timezone"):
            parts.append(f"timezone: {context.extra['timezone']}")
        if context.extra.get("work_dir"):
            parts.append(f"work_dir: {context.extra['work_dir']}")
        if context.extra.get("approval_mode"):
            parts.append(f"approval_mode: {context.extra['approval_mode']}")
        status = context.extra.get("agent_status")
        if isinstance(status, dict) and status:
            parts.append("agent_status:")
            for key in ("phase", "message", "active_tool", "iteration", "max_iterations", "last_event"):
                value = status.get(key)
                if value not in (None, "", 0):
                    parts.append(f"- {key}: {value}")
        if context.current_query:
            parts.append(f"current_query: {context.current_query}")
        if context.recent_observations:
            parts.append("recent_observations:")
            for item in context.recent_observations[-5:]:
                parts.append(f"- {item}")
        parts.append("</runtime_context>")
        return "\n".join(parts)

    def get_cache_key(self, context: LayerContext) -> Optional[str]:
        # Realtime layer is never cached.
        return None
