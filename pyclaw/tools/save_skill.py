import os
from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult

class SaveSkillArgs(BaseModel):
    name: str = Field(..., description="The name of the skill. Must be alphanumeric with underscores/hyphens. Used as directory or file name.")
    description: str = Field(..., description="A short one-line description of the skill (used for indexing).")
    content: str = Field(..., description="The full content of the SKILL.md instructions or the .py tool script.")
    is_python: bool = Field(default=False, description="If True, creates a python tool script (name.py). If False, creates a Markdown skill (name/SKILL.md).")

class SaveSkillTool(BaseTool):
    name = "save_as_skill"
    description = "Persists successful complex procedures or custom Python tools as reusable skills in the local workspace. Use this to permanently expand PyClaw's capabilities."
    args_schema = SaveSkillArgs

    async def execute(self, **kwargs) -> ToolResult:
        name = kwargs.get("name")
        description = kwargs.get("description")
        content = kwargs.get("content")
        is_python = kwargs.get("is_python", False)

        if not name or not content:
            return ToolResult(success=False, content="Error: 'name' and 'content' are required.")

        skills_dir = os.path.abspath(os.path.join(os.getcwd(), "skills"))
        os.makedirs(skills_dir, exist_ok=True)

        # Sanitize name
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', name)
        
        if is_python:
            if not safe_name.endswith('.py'):
                safe_name += '.py'
            target_path = os.path.join(skills_dir, safe_name)
            
            # Security check
            if not target_path.startswith(skills_dir):
                return ToolResult(success=False, content="Error: Invalid skill name.")

            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return ToolResult(
                    success=True, 
                    content=f"✅ SUCCESS: Python tool skill saved to {target_path}.\n"
                            f"The new Python tool will be dynamically loaded in the next turn if valid."
                )
            except Exception as e:
                return ToolResult(success=False, content=f"Failed to save python skill: {str(e)}")
        else:
            target_dir = os.path.join(skills_dir, safe_name)
            # Security check
            if not target_dir.startswith(skills_dir):
                return ToolResult(success=False, content="Error: Invalid skill name.")
            
            os.makedirs(target_dir, exist_ok=True)
            skill_md_path = os.path.join(target_dir, "SKILL.md")
            
            # Prepend the description to the SKILL.md as an H1 or comment if not present
            final_content = content
            if not final_content.startswith("#"):
                final_content = f"# {safe_name}\n> {description}\n\n{final_content}"
                
            try:
                with open(skill_md_path, "w", encoding="utf-8") as f:
                    f.write(final_content)
                return ToolResult(
                    success=True, 
                    content=f"✅ SUCCESS: Markdown skill saved to {skill_md_path}.\n"
                            f"The skill index will be updated automatically. You can activate it using `activate_skill(name=\"{safe_name}\")` in the future."
                )
            except Exception as e:
                return ToolResult(success=False, content=f"Failed to save Markdown skill: {str(e)}")
