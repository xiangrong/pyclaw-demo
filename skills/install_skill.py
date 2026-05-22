import os
import subprocess
from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult

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

        skills_dir = os.path.abspath(os.path.join(os.getcwd(), "skills"))
        os.makedirs(skills_dir, exist_ok=True)

        # Basic repo name extraction
        repo_name = url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
            
        target_dir = os.path.join(skills_dir, repo_name)

        if os.path.exists(target_dir):
            return ToolResult(success=False, content=f"Skill directory already exists: {repo_name}")

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
                content=f"Successfully installed skill repository to {target_dir}.\nGit Output:\n{result.stdout}\n\nYou can now see it in your available skills and activate it if it contains a SKILL.md."
            )
        except subprocess.CalledProcessError as e:
             return ToolResult(success=False, content=f"Failed to clone repository. Git Error:\n{e.stderr}")
        except Exception as e:
            return ToolResult(success=False, content=f"Failed to install skill: {str(e)}")
