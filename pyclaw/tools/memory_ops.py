from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult
from typing import Any

class SaveMemoryArgs(BaseModel):
    text: str = Field(..., description="The knowledge, experience, or fact to save to long-term memory.")
    importance: int = Field(default=3, description="Importance level (1-5). Higher importance may trigger different storage logic in the future.")

class SaveMemoryTool(BaseTool):
    """主动写入语义记忆工具"""
    
    name = "save_memory"
    description = "Proactively save important facts, experiences, or project-specific knowledge to your long-term memory. This helps you learn from interactions and better assist the user in the future."
    args_schema = SaveMemoryArgs

    def __init__(self, semantic_memory: Any):
        self.memory = semantic_memory

    async def execute(self, **kwargs) -> ToolResult:
        text = kwargs.get("text")
        importance = int(kwargs.get("importance", 3))
        
        if not self.memory:
            return ToolResult(success=False, content="Semantic memory is not initialized.")
            
        try:
            metadata = {
                "type": "experience",
                "importance": importance,
                "manual_save": True
            }
            await self.memory.add_memory(text, metadata)
            return ToolResult(success=True, content="Successfully saved experience to long-term memory.")
        except Exception as e:
            return ToolResult(success=False, content=f"Error saving memory: {str(e)}")
