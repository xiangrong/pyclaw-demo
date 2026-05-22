import os
import aiohttp
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from pyclaw.tools.base import BaseTool, ToolResult

class InstallSkillArgs(BaseModel):
    url: str = Field(..., description="The direct URL to the raw Python script file (e.g., raw raw.githubusercontent.com link).")
    filename: str = Field(None, description="Optional. The filename to save as. If not provided, it will be inferred from the URL.")

class InstallSkillTool(BaseTool):
    name = "install_skill"
    description = "Download and install a new Python skill for PyClaw from a given URL."
    args_schema = InstallSkillArgs

    async def execute(self, **kwargs) -> ToolResult:
        url = kwargs.get("url")
        filename = kwargs.get("filename")

        if not url:
            return ToolResult(success=False, content="Error: 'url' parameter is required.")

        # Infer filename from URL if not provided
        if not filename:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.endswith('.py'):
                # fallback generic name if we can't parse a good one
                filename = f"downloaded_skill_{abs(hash(url))}.py"
        
        # Ensure it has a .py extension
        if not filename.endswith('.py'):
            filename += '.py'

        # Use absolute path based on current working directory (set by pyclaw config)
        skills_dir = os.path.abspath(os.path.join(os.getcwd(), "skills"))
        os.makedirs(skills_dir, exist_ok=True)
        file_path = os.path.join(skills_dir, filename)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return ToolResult(
                            success=False, 
                            content=f"Error downloading skill. HTTP Status: {response.status}"
                        )
                    
                    content = await response.text()
                    
                    # Basic validation: check if it looks like a BaseTool (very naive check)
                    if "BaseTool" not in content:
                        return ToolResult(
                            success=False,
                            content="Warning: The downloaded file does not appear to contain a PyClaw BaseTool. Installation aborted to prevent errors."
                        )

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

            return ToolResult(
                success=True, 
                content=f"Successfully installed skill '{filename}' from {url}. The ToolRegistry will automatically hot-load it on the next turn."
            )

        except Exception as e:
            return ToolResult(success=False, content=f"Failed to install skill: {str(e)}")
