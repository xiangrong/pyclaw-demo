from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult
import os

class ListSkillsArgs(BaseModel):
    pass

class ListSkillsTool(BaseTool):
    name = "list_skills"
    description = "Lists all available specialized skills and their short descriptions. Use this to discover what skills are installed."
    args_schema = ListSkillsArgs

    async def execute(self, **kwargs) -> ToolResult:
        # Try to find skills directories
        skills_dirs = [os.path.abspath(os.path.join(os.getcwd(), "skills"))]
        fallback_skills = os.path.expanduser("~/.pyclaw/skills")
        if os.path.exists(fallback_skills):
            abs_fallback = os.path.abspath(fallback_skills)
            if abs_fallback not in skills_dirs:
                skills_dirs.append(abs_fallback)

        skills = []
        for skills_dir in skills_dirs:
            if not os.path.exists(skills_dir):
                continue
            
            # 1. 递归搜索所有带 SKILL.md 的目录
            for root, dirs, files in os.walk(skills_dir):
                if "SKILL.md" in files:
                    skill_md_path = os.path.join(root, "SKILL.md")
                    rel_path = os.path.relpath(root, skills_dir)
                    description = self._extract_description(skill_md_path)
                    if not any(item.startswith(f"- {rel_path}:") for item in skills):
                        skills.append(f"- {rel_path}: {description}")
            
            # 2. 根目录下的独立 .py 技能
            for file in os.listdir(skills_dir):
                if file.endswith(".py") and not file.startswith("__"):
                    skill_name = file[:-3]
                    if not any(item.startswith(f"- {skill_name}:") for item in skills):
                        skills.append(f"- {skill_name}: Python tool script.")
        
        if not skills:
            return ToolResult(success=True, content="No specialized skills found in the available skills directories.")
            
        return ToolResult(success=True, content="Available skills:\n" + "\n".join(sorted(skills)))

    def _extract_description(self, md_path: str) -> str:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    return line[:200]
        except Exception:
            pass
        return "No description available."

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

        # Try to find the skill in available directories
        skills_dirs = [os.path.abspath(os.path.join(os.getcwd(), "skills"))]
        fallback_skills = os.path.expanduser("~/.pyclaw/skills")
        if os.path.exists(fallback_skills):
            skills_dirs.append(os.path.abspath(fallback_skills))

        skill_md_path = None
        for skills_dir in skills_dirs:
            target_path = os.path.abspath(os.path.join(skills_dir, name))
            if not target_path.startswith(skills_dir):
                continue
            
            potential_md = os.path.join(target_path, "SKILL.md")
            if os.path.exists(potential_md):
                skill_md_path = potential_md
                break
        
        if not skill_md_path:
            return ToolResult(success=False, content=f"Skill '{name}' not found or missing SKILL.md in available skill directories.")

        try:
            with open(skill_md_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            wrapped_content = f"<activated_skill name=\"{name}\">\n{content}\n</activated_skill>\n\nCRITICAL INSTRUCTION: You must now strictly follow the procedural guidance and rules provided in the activated skill above for the current task."
            return ToolResult(success=True, content=wrapped_content)
        except Exception as e:
            return ToolResult(success=False, content=f"Failed to read SKILL.md for {name}: {str(e)}")
