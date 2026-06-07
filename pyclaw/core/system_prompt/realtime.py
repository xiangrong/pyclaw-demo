from __future__ import annotations
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
        # Currently, most realtime info is in conversation messages.
        # But we could add things like "Current Time", "System Load", etc.
        return ""

    def get_cache_key(self, context: LayerContext) -> Optional[str]:
        # Realtime layer is never cached.
        return None
