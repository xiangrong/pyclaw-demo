from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Optional

try:
    import lancedb
    import pyarrow as pa
    LANCE_DB_AVAILABLE = True
except ImportError:
    LANCE_DB_AVAILABLE = False
from pydantic import BaseModel

from pyclaw.models.base import BaseModelProvider


class MemoryRecord(BaseModel):
    """一条记忆记录"""
    text: str
    metadata: Dict[str, Any] = {}
    timestamp: str = datetime.now().isoformat()


class SemanticMemory:
    """基于 LanceDB 的语义记忆管理类"""

    @classmethod
    def is_available(cls) -> bool:
        """检查 LanceDB 是否可用"""
        return LANCE_DB_AVAILABLE

    def __init__(
        self,
        model_provider: BaseModelProvider,
        db_path: Optional[str] = None,
        table_name: str = "memories",
    ) -> None:
        self.model = model_provider
        self.db_path = db_path or str(Path.home() / ".config" / "pyclaw" / "lancedb")
        self.table_name = table_name
        self.db = None
        self.table = None
        
        # 确保目录存在
        os.makedirs(self.db_path, exist_ok=True)

    async def _ensure_connected(self) -> None:
        """确保已连接到数据库且表已创建"""
        if not LANCE_DB_AVAILABLE:
            raise ImportError("LanceDB is not installed. Please install it via 'pip install lancedb pyarrow'.")

        if self.db is not None and self.table is not None:
            return

        # LanceDB connect 是同步操作，但在 async 环境下运行良好
        self.db = lancedb.connect(self.db_path)
        
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            
            # 校验维度是否匹配
            try:
                schema = self.table.schema
                vector_field = schema.field("vector")
                # pyarrow 的 FixedSizeListType 包含维度信息
                existing_dim = vector_field.type.list_size
                
                # 获取当前模型的维度
                dummy_vector = await self.model.embed("health check")
                current_dim = len(dummy_vector)
                
                if existing_dim != current_dim:
                    print(f"\n  🚨 [Memory] 维度不匹配! 数据库维度: {existing_dim}, 当前模型维度: {current_dim}")
                    print(f"  🚨 请运行以下命令重置记忆库:\n     rm -rf {self.db_path}\n")
                    # 设置为 None，让后续操作感知到不可用，但不直接崩溃
                    self.table = None
                    return
            except Exception as e:
                print(f"  ⚠️  校验维度时出错: {e}")
        else:
            # 创建初始表结构
            # 动态获取 embedding 维度
            print(f"🔍 [Memory] Initializing table '{self.table_name}' with dynamic dimension...")
            dummy_vector = await self.model.embed("health check")
            dim = len(dummy_vector)
            
            schema = pa.schema([
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()), # JSON 字符串
                pa.field("timestamp", pa.string()),
            ])
            self.table = self.db.create_table(self.table_name, schema=schema)

    async def add_memory(self, text: str, metadata: Dict[str, Any] = {}) -> None:
        """添加一条记忆"""
        await self._ensure_connected()
        
        import json
        vector = await self.model.embed(text)
        
        record = {
            "vector": vector,
            "text": text,
            "metadata": json.dumps(metadata),
            "timestamp": datetime.now().isoformat(),
        }
        
        # LanceDB add 也是同步阻塞的，在 async 中使用建议 wrap 或直接执行（通常很快）
        if self.table is not None:
            self.table.add([record])
        else:
            print("  ⚠️  Memory table is not initialized, skipping add_memory.")

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相关记忆"""
        await self._ensure_connected()
        
        import json
        query_vector = await self.model.embed(query)
        
        # 执行向量搜索，明确指定向量列名
        try:
            results = self.table.search(query_vector, vector_column_name="vector").limit(limit).to_list()
        except Exception as e:
            # 如果搜索失败（可能是维度不匹配），提示用户
            err_str = str(e).lower()
            if "dimension" in err_str or "vector column" in err_str:
                print(f"  ⚠️  语义记忆搜索失败 (可能是模型变更导致维度不匹配): {e}")
                print(f"  💡 提示: 如果您更换了 Embedding 模型，请尝试删除 {self.db_path} 目录后重试。")
                return []
            raise e
        
        formatted_results = []
        for r in results:
            formatted_results.append({
                "text": r["text"],
                "metadata": json.loads(r["metadata"]),
                "timestamp": r["timestamp"],
                "score": r.get("_distance", 0.0), # 距离分数
            })
            
        return formatted_results

    async def add_session_interaction(self, user_msg: str, assistant_msg: str, session_id: str) -> None:
        """记录一次会话交互"""
        text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        metadata = {
            "type": "interaction",
            "session_id": session_id,
        }
        await self.add_memory(text, metadata)
