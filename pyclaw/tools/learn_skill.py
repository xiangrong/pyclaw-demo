from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field
import trafilatura

from .base import BaseTool, ToolResult
from .web_extract import unsafe_url_reason


class LearnFromDocArgs(BaseModel):
    source: str = Field(..., description="The URL of the documentation or a local file path.")
    skill_name: str = Field(..., description="The name for the new skill (e.g., 'weather', 'stripe_api'). Use snake_case.")


class LearnFromDocTool(BaseTool):
    """从文档中自主学习并创建新技能"""

    name = "learn_skill_from_doc"
    description = (
        "Read documentation from a URL or local file, analyze its API/functionality, "
        "and autonomously generate a new functional skill (SKILL.md + optional scripts) "
        "in the skills/ directory. Use this when you need to learn how to use a new tool or API."
    )
    args_schema = LearnFromDocArgs

    def __init__(self, agent_instance: Any):
        self.agent = agent_instance

    async def execute(self, source: str, skill_name: str) -> ToolResult:
        try:
            # 1. 获取文档内容
            content = ""
            if source.startswith(("http://", "https://")):
                unsafe_reason = unsafe_url_reason(source)
                if unsafe_reason:
                    return ToolResult(
                        success=False,
                        content=f"Unsafe or unsupported URL '{source}': {unsafe_reason}",
                        error_code="unsafe_url",
                        requires_model_repair=True,
                    )
                print(f"  📥 [LearnSkill] Fetching documentation from URL: {source}")
                def _fetch():
                    downloaded = trafilatura.fetch_url(source)
                    return trafilatura.extract(downloaded) if downloaded else None
                
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(None, _fetch)
            else:
                print(f"  📥 [LearnSkill] Reading documentation from file: {source}")
                try:
                    resolved = self.validate_path(source)
                except PermissionError as exc:
                    return ToolResult(
                        success=False,
                        content=f"Access denied while reading documentation: {exc}",
                        error_code="path_outside_workspace",
                        requires_model_repair=True,
                    )
                path = Path(resolved)
                if path.exists():
                    content = path.read_text(encoding="utf-8")
                else:
                    return ToolResult(success=False, content=f"Error: File not found at {source}")

            if not content:
                return ToolResult(success=False, content=f"Error: Could not extract content from {source}")

            # 2. 调用模型生成技能
            print(f"  ⚙️ [LearnSkill] Analyzing documentation and generating skill '{skill_name}'...")
            
            meta_prompt = (
                f"你是一名资深的「AI 技能工程师」。你的任务是根据提供的文档，为 PyClaw Agent 编写一个新的技能包。\n\n"
                f"技能名称：{skill_name}\n\n"
                "要求：\n"
                "1. 理解文档中描述的 API、命令行工具或逻辑流程。\n"
                "2. 编写 `SKILL.md`：详细描述该技能的用途、工作流以及 Agent 应该如何通过终端命令（如 curl, python 脚本等）来调用它。\n"
                "3. 如果 API 调用比较复杂，请额外编写一个辅助 Python 脚本（如 `{skill_name}_wrapper.py`）。\n"
                "4. 遵循「渐进式能力披露」原则，SKILL.md 应作为核心引导。\n\n"
                "请严格按照以下格式输出：\n"
                "---SKILL.MD---\n(这里是 SKILL.md 的内容)\n"
                "---SCRIPT---\n(这里是辅助脚本的内容，如果没有则留空)\n"
                "---END---\n"
            )

            messages = [
                {"role": "system", "content": meta_prompt},
                {"role": "user", "content": f"文档内容如下：\n\n{content[:15000]}"} # 截断避免溢出
            ]

            response = await self.agent.model.chat(messages=messages, tools=None)
            
            # 3. 解析输出
            skill_md_match = re.search(r"---SKILL\.MD---\n(.*?)(?=\n---)", response, re.DOTALL)
            script_match = re.search(r"---SCRIPT---\n(.*?)(?=\n---)", response, re.DOTALL)
            
            skill_md_content = skill_md_match.group(1).strip() if skill_md_match else ""
            script_content = script_match.group(1).strip() if script_match else ""

            if not skill_md_content:
                return ToolResult(success=False, content="Error: Model failed to generate SKILL.md content.")

            # 4. 保存文件
            safe_skill_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", skill_name).strip("_") or "learned_skill"
            work_root = Path(self.agent.work_dir).resolve()
            skills_dir = work_root / "skills" / safe_skill_name
            if os.path.commonpath([str(skills_dir.resolve().parent.parent), str(work_root)]) != str(work_root):
                return ToolResult(success=False, content="Error: Invalid skill path.", error_code="invalid_skill_path")
            skills_dir.mkdir(parents=True, exist_ok=True)
            
            (skills_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
            review_path = ""
            
            if script_content and len(script_content) > 10:
                script_path = skills_dir / f"{safe_skill_name}_wrapper.py.review"
                script_path.write_text(script_content, encoding="utf-8")
                review_path = str(script_path)
                print(f"  ✅ [LearnSkill] Wrapper script saved to {script_path}")

            print(f"  ✅ [LearnSkill] Skill '{safe_skill_name}' successfully learned and saved to {skills_dir}")

            return ToolResult(
                success=True,
                content=f"SUCCESS: Skill '{safe_skill_name}' has been learned from {source}.\n"
                        f"The new skill is located at {skills_dir}.\n"
                        "If a helper script was generated it was saved as .py.review and is not executable until human review adds '# pyclaw: trusted-skill' and renames it to .py.\n"
                        f"IMPORTANT: You can now see it in the <available_skills> index and use it by calling `activate_skill(name='{safe_skill_name}')`.",
                metadata={
                    "skill_name": safe_skill_name,
                    "skill_dir": str(skills_dir),
                    "script_review_path": review_path,
                    "review_required": bool(review_path),
                    "trust_level": "untrusted_skill",
                },
            )

        except Exception as e:
            return ToolResult(success=False, content=f"Error learning skill: {str(e)}")
