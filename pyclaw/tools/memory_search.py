from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult
from typing import Any

class MemorySearchArgs(BaseModel):
    query: str = Field(..., description="The search query to find relevant past interactions or knowledge.")
    limit: int = Field(default=5, description="Maximum number of relevant memories to return.")

class MemorySearchTool(BaseTool):
    """主动搜索语义记忆工具"""
    
    name = "search_memory"
    description = "Proactively search your long-term semantic memory for relevant past conversations, decisions, or facts that might be useful for the current task."
    args_schema = MemorySearchArgs

    def __init__(self, semantic_memory: Any):
        self.memory = semantic_memory

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query")
        limit = int(kwargs.get("limit", 5))
        
        if not self.memory:
            return ToolResult(success=False, content="Semantic memory is not initialized.")
            
        try:
            results = await self.memory.search(query, limit=limit)
            if not results:
                return ToolResult(success=True, content="No relevant memories found for this query.")
                
            output = "Found the following relevant memories:\n\n"
            for r in results:
                output += f"--- Memory ({r['timestamp']}, Score: {r['score']:.4f}) ---\n{r['text']}\n\n"
                
            return ToolResult(success=True, content=output)
        except Exception as e:
            return ToolResult(success=False, content=f"Error searching memory: {str(e)}")
