from __future__ import annotations

from typing import Any, AsyncGenerator, List, Optional
from .base import BaseModelProvider

class LocalEmbeddingProvider(BaseModelProvider):
    """本地模型提供商 - 仅用于生成 Embedding"""

    name = "local"

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu"
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"📡 [LocalEmbedding] Loading model {self.model_name} onto {self.device}...")
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError("Please install 'sentence-transformers' to use local embeddings: pip install sentence-transformers")
        return self._model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("LocalEmbeddingProvider only supports embeddings, not chat.")

    def format_tool_def(self, tool_def: dict[str, Any]) -> dict[str, Any]:
        return tool_def

    async def embed(self, text: str) -> list[float]:
        """本地生成嵌入向量"""
        model = self._load_model()
        import asyncio
        # 将同步推理放到线程池中执行，避免阻塞事件循环
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, lambda: model.encode(text, normalize_embeddings=True))
        return vector.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """本地批量生成嵌入向量"""
        if not texts:
            return []
        model = self._load_model()
        import asyncio
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, lambda: model.encode(texts, normalize_embeddings=True))
        return vectors.tolist()
