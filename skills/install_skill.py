import os
import subprocess
from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult

class UninstallSkillArgs(BaseModel):
    name: str = Field(..., description="The name of the skill directory to uninstall.")

class UninstallSkillTool(BaseTool):
    name = "uninstall_skill"
    description = "Safely removes a skill directory from the PyClaw workspace."
    args_schema = UninstallSkillArgs

    async def execute(self, **kwargs) -> ToolResult:
        name = kwargs.get("name")
        if not name:
            return ToolResult(success=False, content="Error: 'name' parameter is required.")

        skills_dir = os.path.abspath(os.path.join(os.getcwd(), "skills"))
        target_dir = os.path.abspath(os.path.join(skills_dir, name))

        # Security check
        if not target_dir.startswith(skills_dir) or target_dir == skills_dir:
            return ToolResult(success=False, content="Error: Invalid skill name or access denied.")

        if not os.path.exists(target_dir):
            return ToolResult(success=False, content=f"Error: Skill '{name}' not found at {target_dir}")

        try:
            import shutil
            shutil.rmtree(target_dir)
            return ToolResult(success=True, content=f"SUCCESS: Skill '{name}' has been uninstalled and its directory removed.")
        except Exception as e:
            return ToolResult(success=False, content=f"Failed to uninstall skill: {str(e)}")

class InstallSkillArgs(BaseModel):
    url: str = Field(..., description="The git repository URL of the skill to install.")

class InstallSkillTool(BaseTool):
    name = "install_skill"
    description = "Downloads and installs a new skill repository into the PyClaw workspace using git clone."
    args_schema = InstallSkillArgs

    async def execute(self, **kwargs) -> ToolResult:
        url = kwargs.get("url")

        if not url:
            return ToolResult(success=False, content="Error: 'url' parameter is required.")

        # Always install into the current work_dir/skills
        skills_dir = os.path.abspath(os.path.join(os.getcwd(), "skills"))
        os.makedirs(skills_dir, exist_ok=True)

        # Basic repo name extraction
        repo_name = url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
            
        target_dir = os.path.join(skills_dir, repo_name)

        if os.path.exists(target_dir):
            return ToolResult(success=False, content=f"Skill directory already exists at {target_dir}. If you want to update it, please use the terminal to pull changes.")

        try:
            # Use subprocess to run git clone
            result = subprocess.run(
                ["git", "clone", url, target_dir],
                capture_output=True,
                text=True,
                check=True
            )
            return ToolResult(
                success=True, 
                content=f"SUCCESS: Skill repository installed to {target_dir}.\n"
                        f"IMPORTANT: The new skill is now available. You can find it by calling `list_skills` or by checking the <available_skills> index in your next turn. "
                        f"If it contains a SKILL.md, remember to call `activate_skill(name=\"{repo_name}\")` before using it."
            )
        except subprocess.CalledProcessError as e:
             return ToolResult(success=False, content=f"Failed to clone repository. Git Error:\n{e.stderr}")
        except Exception as e:
            return ToolResult(success=False, content=f"Failed to install skill: {str(e)}")
