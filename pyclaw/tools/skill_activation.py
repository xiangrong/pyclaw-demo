import hashlib
import os
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, Field

from pyclaw.tools.base import BaseTool, ToolResult


@dataclass(frozen=True)
class SkillCandidate:
    """A discoverable markdown skill."""

    root_dir: str
    rel_path: str
    skill_md_path: str
    frontmatter_name: str | None = None
    description: str = "No description available."


def _available_skills_dirs() -> list[str]:
    """Return skill roots in lookup order, de-duplicated."""
    roots = [os.path.abspath(os.path.join(os.getcwd(), "skills"))]
    fallback_skills = os.path.expanduser("~/.pyclaw/skills")
    if os.path.exists(fallback_skills):
        roots.append(os.path.abspath(fallback_skills))

    deduped: list[str] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return deduped


def _is_path_within(path: str, root: str) -> bool:
    """Return True when path is inside root."""
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _clean_frontmatter_value(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1].strip()
    return value


def _extract_frontmatter_fields(md_path: str) -> dict[str, str]:
    """Extract a tiny YAML-frontmatter subset without adding a dependency."""
    fields: dict[str, str] = {}
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            lines = []
            for _ in range(120):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
    except Exception:
        return fields

    if not lines or lines[0].strip() != "---":
        return fields

    i = 1
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "---":
            break
        if not stripped or stripped.startswith("#") or ":" not in line:
            i += 1
            continue

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value in {">", ">-", "|", "|-"}:
            block: list[str] = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                if next_stripped == "---":
                    i -= 1
                    break
                if next_line and not next_line.startswith((" ", "\t")) and ":" in next_line:
                    i -= 1
                    break
                if next_stripped:
                    block.append(next_stripped)
                i += 1
            fields[key] = " ".join(block).strip()
        else:
            fields[key] = _clean_frontmatter_value(value)
        i += 1

    return fields


def _extract_description(md_path: str) -> str:
    fields = _extract_frontmatter_fields(md_path)
    description = fields.get("description", "").strip()
    if description:
        return description[:200]
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            in_frontmatter = False
            for line in f:
                stripped = line.strip()
                if stripped == "---":
                    in_frontmatter = not in_frontmatter
                    continue
                if in_frontmatter or not stripped or stripped.startswith("#"):
                    continue
                return stripped[:200]
    except Exception:
        pass
    return "No description available."


def _discover_markdown_skills(skills_dirs: Iterable[str]) -> list[SkillCandidate]:
    candidates: list[SkillCandidate] = []
    seen_paths: set[str] = set()
    for skills_dir in skills_dirs:
        if not os.path.exists(skills_dir):
            continue
        for root, _dirs, files in os.walk(skills_dir):
            if "SKILL.md" not in files:
                continue
            skill_md_path = os.path.join(root, "SKILL.md")
            abs_skill_md_path = os.path.abspath(skill_md_path)
            if abs_skill_md_path in seen_paths:
                continue
            seen_paths.add(abs_skill_md_path)
            rel_path = os.path.relpath(root, skills_dir)
            fields = _extract_frontmatter_fields(skill_md_path)
            candidates.append(
                SkillCandidate(
                    root_dir=skills_dir,
                    rel_path=rel_path,
                    skill_md_path=skill_md_path,
                    frontmatter_name=fields.get("name") or None,
                    description=(fields.get("description") or _extract_description(skill_md_path))[:200],
                )
            )
    return candidates


def _normalize_skill_lookup_name(name: str) -> str:
    return name.strip().strip("/")


def resolve_markdown_skill(name: str, skills_dirs: Iterable[str] | None = None) -> SkillCandidate | None:
    """Resolve a markdown skill by path, frontmatter name, or unique basename.

    This supports installed skill bundles that expose a user-facing frontmatter
    name (for example ``baoyu-design``) while their actual SKILL.md lives under
    a nested path such as ``baoyu-design/skills/baoyu-design``.
    """
    normalized = _normalize_skill_lookup_name(name)
    if not normalized or os.path.isabs(normalized) or ".." in normalized.split(os.sep):
        return None

    roots = list(skills_dirs) if skills_dirs is not None else _available_skills_dirs()

    # 1. Exact path match keeps backwards compatibility with existing calls.
    for skills_dir in roots:
        target_path = os.path.abspath(os.path.join(skills_dir, normalized))
        if not _is_path_within(target_path, skills_dir):
            continue
        potential_md = os.path.join(target_path, "SKILL.md")
        if os.path.exists(potential_md):
            fields = _extract_frontmatter_fields(potential_md)
            return SkillCandidate(
                root_dir=skills_dir,
                rel_path=os.path.relpath(target_path, skills_dir),
                skill_md_path=potential_md,
                frontmatter_name=fields.get("name") or None,
                description=(fields.get("description") or _extract_description(potential_md))[:200],
            )

    candidates = _discover_markdown_skills(roots)

    # 2. Exact frontmatter name (what users and models naturally use).
    frontmatter_matches = [c for c in candidates if c.frontmatter_name == normalized]
    if len(frontmatter_matches) == 1:
        return frontmatter_matches[0]

    # Prefer the shortest path when a bundle accidentally ships duplicate copies
    # with the same declared name.
    if len(frontmatter_matches) > 1:
        return sorted(frontmatter_matches, key=lambda c: (c.rel_path.count(os.sep), len(c.rel_path)))[0]

    # 3. Unique directory basename fallback.
    basename_matches = [c for c in candidates if os.path.basename(c.rel_path) == normalized]
    if len(basename_matches) == 1:
        return basename_matches[0]
    if len(basename_matches) > 1:
        return sorted(basename_matches, key=lambda c: (c.rel_path.count(os.sep), len(c.rel_path)))[0]

    return None

class ListSkillsArgs(BaseModel):
    pass

class ListSkillsTool(BaseTool):
    name = "list_skills"
    description = "Lists all available specialized skills and their short descriptions. Use this to discover what skills are installed."
    args_schema = ListSkillsArgs

    async def execute(self, **kwargs) -> ToolResult:
        skills_dirs = _available_skills_dirs()

        skills = []
        for skills_dir in skills_dirs:
            if not os.path.exists(skills_dir):
                continue
            
            # 1. 递归搜索所有带 SKILL.md 的目录
            for candidate in _discover_markdown_skills([skills_dir]):
                display_name = candidate.frontmatter_name or candidate.rel_path
                path_hint = "" if display_name == candidate.rel_path else f" (path: {candidate.rel_path})"
                if not any(item.startswith(f"- {display_name}:") for item in skills):
                    skills.append(f"- {display_name}: {candidate.description}{path_hint}")
            
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
        return _extract_description(md_path)

class ActivateSkillArgs(BaseModel):
    name: str = Field(..., description="The name of the skill directory to activate.")

class ActivateSkillTool(BaseTool):
    name = "activate_skill"
    description = "Activates a specialized skill by reading its SKILL.md file. Use this when you identify a task that matches an available skill's description."
    args_schema = ActivateSkillArgs

    async def execute(self, **kwargs) -> ToolResult:
        name = _normalize_skill_lookup_name(str(kwargs.get("name") or ""))
        if not name:
            return ToolResult(success=False, content="Skill name is required.")

        # Try to find the skill in available directories
        skills_dirs = _available_skills_dirs()

        # Check for Python tool script first
        py_tool_path = None
        for skills_dir in skills_dirs:
            target_path = os.path.abspath(os.path.join(skills_dir, f"{name}.py"))
            if target_path.startswith(skills_dir) and os.path.exists(target_path):
                py_tool_path = target_path
                break
                
        if py_tool_path:
            return ToolResult(
                success=True, 
                content=f"✅ Python tool skill '{name}' activated successfully. The tool specification is now available for you to use in this session. You can see its signature in your next turn.",
                metadata={"activated_skill": name}
            )

        # Then check for SKILL.md
        candidate = resolve_markdown_skill(name, skills_dirs)
        if not candidate:
            return ToolResult(success=False, content=f"Skill '{name}' not found or missing SKILL.md in available skill directories.")

        try:
            with open(candidate.skill_md_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            activated_name = candidate.frontmatter_name or candidate.rel_path
            abs_skill_md_path = os.path.abspath(candidate.skill_md_path)
            abs_root_dir = os.path.abspath(candidate.root_dir)
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            wrapped_content = f"<activated_skill name=\"{activated_name}\" path=\"{candidate.rel_path}\" skill_md_path=\"{abs_skill_md_path}\" root_dir=\"{os.path.dirname(abs_skill_md_path)}\">\n{content}\n</activated_skill>\n\nCRITICAL INSTRUCTION: You must now strictly follow the procedural guidance and rules provided in the activated skill above for the current task. Load any referenced files relative to root_dir."
            return ToolResult(
                success=True,
                content=wrapped_content,
                metadata={
                    "activated_skill": activated_name,
                    "activated_skill_path": candidate.rel_path,
                    "activated_skill_md_path": abs_skill_md_path,
                    "activated_skill_root_dir": abs_root_dir,
                    "activated_skill_content_sha256": content_hash,
                    "activated_skill_description": candidate.description,
                },
            )
        except Exception as e:
            return ToolResult(success=False, content=f"Failed to read SKILL.md for {name}: {str(e)}")
