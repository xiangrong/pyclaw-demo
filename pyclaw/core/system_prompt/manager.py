from __future__ import annotations
from typing import Dict, List, Optional, Type
import hashlib

from .models import LayerContext, SystemPromptConfig
from .base import BaseLayer
from .static import StaticLayer
from .session import SessionLayer
from .realtime import RealtimeLayer


class SystemPromptManager:
    """
    Manager for the three-layer system prompt architecture.
    Handles rendering, caching, and assembly.
    """

    def __init__(self, config: Optional[SystemPromptConfig] = None) -> None:
        self.config = config or SystemPromptConfig()
        self.layers: List[BaseLayer] = [
            StaticLayer(),
            SessionLayer(),
            RealtimeLayer()
        ]
        # Memory cache for rendered layers
        self._cache: Dict[str, str] = {}
        # Track hash of context fields to detect changes for session layer
        self._context_hashes: Dict[str, str] = {}

    async def generate_prompt(self, context: LayerContext) -> str:
        """
        Generate the full system prompt by rendering and joining all layers.
        """
        layer_contents = []
        
        for layer in self.layers:
            cache_key = layer.get_cache_key(context)
            
            # Check if we should use cache
            use_cache = False
            if self.config.enable_cache and cache_key:
                # For SessionLayer, we need to check if objective/plan/rag changed
                if isinstance(layer, SessionLayer):
                    current_hash = self._get_session_context_hash(context)
                    if self._context_hashes.get(cache_key) == current_hash:
                        use_cache = True
                    else:
                        self._context_hashes[cache_key] = current_hash
                else:
                    use_cache = True

            if use_cache and cache_key in self._cache:
                content = self._cache[cache_key]
            else:
                content = await layer.render(context)
                if cache_key:
                    self._cache[cache_key] = content
            
            if content:
                layer_contents.append(content)

        return "\n\n".join(layer_contents)

    def _get_session_context_hash(self, context: LayerContext) -> str:
        """
        Create a hash of session-specific context fields to detect changes.
        """
        data = "|".join([
            context.current_objective,
            context.current_plan,
            context.semantic_memory,
            context.experience_memory,
            context.coding_task_status,
            context.active_skills_context,
            # Explicit deliverable/skill tasks inject a controller-owned
            # workspace adapter into the SessionLayer.  This field changes
            # when a new file-deliverable contract is inferred (artifact dir,
            # required skill docs, original task).  If it is not part of the
            # cache hash, the model can receive a stale session prompt from a
            # previous turn and miss the Hermes/OpenClaw-style skill workflow
            # contract entirely.
            context.deliverable_workspace_context,
        ])
        return hashlib.md5(data.encode()).hexdigest()

    def invalidate_static_cache(self) -> None:
        """
        Invalidate the static layer cache (e.g. when skills are updated).
        """
        key = "global_static_layer"
        if key in self._cache:
            del self._cache[key]
            
    def invalidate_session_cache(self, session_id: str) -> None:
        """
        Invalidate cache for a specific session.
        """
        key = f"session_layer_{session_id}"
        if key in self._cache:
            del self._cache[key]
        if key in self._context_hashes:
            del self._context_hashes[key]
