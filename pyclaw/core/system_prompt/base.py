from __future__ import annotations
from abc import ABC, abstractmethod
from .models import LayerContext


class BaseLayer(ABC):
    """
    Base class for all system prompt layers.
    """
    
    @abstractmethod
    async def render(self, context: LayerContext) -> str:
        """
        Render the layer content as a string.
        """
        pass

    def get_cache_key(self, context: LayerContext) -> Optional[str]:
        """
        Return a cache key for this layer. Return None to disable caching for this layer.
        """
        return None
