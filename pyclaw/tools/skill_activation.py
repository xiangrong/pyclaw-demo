from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult
import os

class ActivateSkillArgs(BaseModel):
    name: str = Field(..., description="The name of the skill directory to activate.")

class ActivateSkillTool(BaseTool):
    name = "activate_skill"
    description = "Activates a specialized skill by reading its SKILL.md file. Use this when you identify a task that matches an available skill's description."
    args_schema = ActivateSkillArgs

    async def execute(self, **kwargs) -> ToolResult:
        name = kwargs.get("name")
        if not name:
            return ToolResult(success=False, content="Skill name is required.")

        # Resolve path safely within the work_dir/skills folder
        skills_dir = os.path.abspath(os.path.join(os.getcwd(), "skills"))
        skill_path = os.path.join(skills_dir, name)
        skill_md_path = os.path.join(skill_path, "SKILL.md")

        if not os.path.exists(skill_md_path):
            return ToolResult(success=False, content=f"Skill '{name}' not found or missing SKILL.md. Path checked: {skill_md_path}")

        try:
            with open(skill_md_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            wrapped_content = f"<activated_skill name=\"{name}\">\n{content}\n</activated_skill>\n\nCRITICAL INSTRUCTION: You must now strictly follow the procedural guidance and rules provided in the activated skill above for the current task."
            return ToolResult(success=True, content=wrapped_content)
        except Exception as e:
            return ToolResult(success=False, content=f"Failed to read SKILL.md for {name}: {str(e)}")
