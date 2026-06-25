from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult


EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".jar",
    ".war",
    ".class",
    ".dex",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".mp3",
    ".mp4",
    ".mov",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
}

SYMBOL_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".kt",
    ".kts",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".cxx",
    ".hpp",
    ".cs",
    ".swift",
    ".rb",
    ".php",
}


class GrepCodeArgs(BaseModel):
    pattern: str = Field(description="Regex or literal text to search for")
    path: str = Field(default=".", description="File or directory to search within, relative to work_dir by default")
    regex: bool = Field(default=True, description="Treat pattern as a regular expression when true; literal substring when false")
    case_sensitive: bool = Field(default=True, description="Whether matching is case-sensitive")
    include: str | None = Field(default=None, description="Optional file-name regex filter, e.g. '\\.py$' or 'MainActivity\\.java$'")
    context_lines: int = Field(default=0, ge=0, le=5, description="Number of context lines before and after each match")
    max_matches: int = Field(default=50, ge=1, le=200, description="Maximum matches to return")
    max_chars: int = Field(default=12000, ge=1000, le=50000, description="Maximum characters in the response")


class ReadLinesArgs(BaseModel):
    path: str = Field(description="File path to read")
    start_line: int = Field(ge=1, description="1-based first line to read")
    end_line: int = Field(ge=1, description="1-based last line to read, inclusive")
    show_line_numbers: bool = Field(default=True, description="Prefix each returned line with its line number")
    max_chars: int = Field(default=12000, ge=1000, le=50000, description="Maximum characters in the response")


class ListSymbolsArgs(BaseModel):
    path: str = Field(default=".", description="File or directory to scan, relative to work_dir by default")
    include: str | None = Field(default=None, description="Optional file-name regex filter, e.g. '\\.py$' or 'MainActivity\\.java$'")
    include_private: bool = Field(default=False, description="Include private/internal symbols such as _name")
    max_symbols: int = Field(default=200, ge=1, le=1000, description="Maximum symbols to return")
    max_chars: int = Field(default=12000, ge=1000, le=50000, description="Maximum characters in the response")


class FindRefsArgs(BaseModel):
    symbol: str = Field(description="Identifier or dotted symbol to find references for, e.g. collectDeviceInfo or Foo.bar")
    path: str = Field(default=".", description="File or directory to search within, relative to work_dir by default")
    include: str | None = Field(default=None, description="Optional file-name regex filter, e.g. '\\.py$' or 'MainActivity\\.java$'")
    include_definitions: bool = Field(default=False, description="Include likely definition lines in addition to references")
    context_lines: int = Field(default=1, ge=0, le=5, description="Number of context lines before and after each reference")
    max_matches: int = Field(default=80, ge=1, le=300, description="Maximum references to return")
    max_chars: int = Field(default=16000, ge=1000, le=60000, description="Maximum characters in the response")


class GotoDefArgs(BaseModel):
    symbol: str = Field(description="Identifier or dotted symbol to locate the definition for, e.g. collectDeviceInfo or Foo.bar")
    path: str = Field(default=".", description="File or directory to search within, relative to work_dir by default")
    include: str | None = Field(default=None, description="Optional file-name regex filter, e.g. '\\.py$' or 'MainActivity\\.java$'")
    context_lines: int = Field(default=4, ge=0, le=20, description="Number of surrounding lines to show around each definition")
    max_matches: int = Field(default=20, ge=1, le=100, description="Maximum likely definitions to return")
    max_chars: int = Field(default=16000, ge=1000, le=60000, description="Maximum characters in the response")


@dataclass(frozen=True)
class Symbol:
    file: str
    line: int
    kind: str
    name: str
    signature: str = ""


class CodeSearchMixin:
    def _safe_root(self, path: str) -> Path:
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded) and self.work_dir:
            expanded = os.path.join(self.work_dir, expanded)
        return Path(self.validate_path(expanded))

    def _display_path(self, path: Path) -> str:
        try:
            if self.work_dir:
                return str(path.resolve().relative_to(Path(self.work_dir).resolve()))
        except ValueError:
            pass
        return str(path)

    def _iter_text_files(self, root: Path, include: str | None = None) -> Iterable[Path]:
        include_re = re.compile(include) if include else None
        if root.is_file():
            if self._is_probably_text(root) and (include_re is None or include_re.search(root.name)):
                yield root
            return

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
            for filename in filenames:
                path = Path(dirpath) / filename
                if include_re is not None and not include_re.search(str(path)):
                    continue
                if self._is_probably_text(path):
                    yield path

    def _is_probably_text(self, path: Path) -> bool:
        if path.suffix.lower() in BINARY_EXTENSIONS:
            return False
        try:
            with path.open("rb") as f:
                chunk = f.read(2048)
        except OSError:
            return False
        return b"\x00" not in chunk

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... output truncated; narrow path/pattern or lower max_matches ..."

    def _identifier_tail(self, symbol: str) -> str:
        tail = symbol.strip().split(".")[-1]
        return tail.strip()

    def _identifier_re(self, symbol: str) -> re.Pattern[str]:
        tail = re.escape(self._identifier_tail(symbol))
        return re.compile(rf"(?<![A-Za-z0-9_$]){tail}(?![A-Za-z0-9_$])")

    def _render_line_range(
        self,
        *,
        path: Path,
        lines: list[str],
        center_line: int,
        context_lines: int,
        marker: str = ":",
    ) -> list[str]:
        display = self._display_path(path)
        start = max(1, center_line - context_lines)
        end = min(len(lines), center_line + context_lines)
        rendered: list[str] = []
        for current in range(start, end + 1):
            sep = marker if current == center_line else "-"
            rendered.append(f"{display}{sep}{current}: {lines[current - 1]}")
        return rendered





class GrepCodeTool(CodeSearchMixin, BaseTool):
    name = "grep_code"
    description = (
        "Search code with line numbers and optional context. Prefer this over reading whole large files "
        "when locating symbols, call sites, TODOs, errors, or implementation branches."
    )
    args_schema = GrepCodeArgs

    async def execute(self, **kwargs: object) -> ToolResult:
        pattern = str(kwargs.get("pattern", ""))
        search_path = str(kwargs.get("path", "."))
        regex = bool(kwargs.get("regex", True))
        case_sensitive = bool(kwargs.get("case_sensitive", True))
        include = kwargs.get("include")
        context_lines = int(kwargs.get("context_lines", 0))
        max_matches = int(kwargs.get("max_matches", 50))
        max_chars = int(kwargs.get("max_chars", 12000))

        if not pattern:
            return ToolResult(success=False, content="Error: pattern must not be empty")

        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            matcher = re.compile(pattern if regex else re.escape(pattern), flags)
            root = self._safe_root(search_path)
            matches: list[str] = []
            total = 0

            for path in self._iter_text_files(root, str(include) if include else None):
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                matched_lines = [idx for idx, line in enumerate(lines, start=1) if matcher.search(line)]
                for line_no in matched_lines:
                    total += 1
                    if len(matches) >= max_matches:
                        continue
                    display = self._display_path(path)
                    start = max(1, line_no - context_lines)
                    end = min(len(lines), line_no + context_lines)
                    for current in range(start, end + 1):
                        marker = ":" if current == line_no else "-"
                        matches.append(f"{display}{marker}{current}: {lines[current - 1]}")
                    if context_lines:
                        matches.append("--")

            if not matches:
                return ToolResult(success=True, content=f"No matches for pattern: {pattern}")

            header = f"Found {total} match(es) for {pattern!r}; showing {min(total, max_matches)}."
            return ToolResult(success=True, content=self._truncate(header + "\n" + "\n".join(matches), max_chars))
        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
        except re.error as e:
            return ToolResult(success=False, content=f"Invalid regex: {e}")
        except Exception as e:
            return ToolResult(success=False, content=f"Error searching code: {type(e).__name__}: {e}")


class ReadLinesTool(CodeSearchMixin, BaseTool):
    name = "read_lines"
    description = "Read a precise 1-based line range from a file. Prefer this over read_file for large source files."
    args_schema = ReadLinesArgs

    async def execute(self, **kwargs: object) -> ToolResult:
        path = str(kwargs.get("path", ""))
        start_line = int(kwargs.get("start_line", 1))
        end_line = int(kwargs.get("end_line", 1))
        show_line_numbers = bool(kwargs.get("show_line_numbers", True))
        max_chars = int(kwargs.get("max_chars", 12000))

        if end_line < start_line:
            return ToolResult(success=False, content="Error: end_line must be greater than or equal to start_line")

        try:
            safe_path = self._safe_root(path)
            lines = safe_path.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            selected = lines[start_line - 1:min(end_line, total)]
            rendered = []
            for offset, line in enumerate(selected, start=start_line):
                rendered.append(f"{offset:>6} | {line}" if show_line_numbers else line)
            header = f"File: {path} lines {start_line}-{min(end_line, total)} of {total}"
            return ToolResult(success=True, content=self._truncate(header + "\n" + "\n".join(rendered), max_chars))
        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
        except FileNotFoundError:
            return ToolResult(success=False, content=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, content=f"Error reading lines: {type(e).__name__}: {e}")


class ListSymbolsTool(CodeSearchMixin, BaseTool):
    name = "list_symbols"
    description = (
        "List top-level and nested code symbols with file and line numbers. Use this to map a codebase before "
        "opening implementation ranges. Supports Python AST and regex-based extraction for common languages."
    )
    args_schema = ListSymbolsArgs

    async def execute(self, **kwargs: object) -> ToolResult:
        scan_path = str(kwargs.get("path", "."))
        include = kwargs.get("include")
        include_private = bool(kwargs.get("include_private", False))
        max_symbols = int(kwargs.get("max_symbols", 200))
        max_chars = int(kwargs.get("max_chars", 12000))

        try:
            root = self._safe_root(scan_path)
            symbols: list[Symbol] = []
            for path in self._iter_text_files(root, str(include) if include else None):
                if path.suffix.lower() not in SYMBOL_EXTENSIONS:
                    continue
                symbols.extend(self._symbols_for_file(path, include_private))
                if len(symbols) >= max_symbols:
                    break

            if not symbols:
                return ToolResult(success=True, content=f"No symbols found under: {scan_path}")

            symbols = sorted(symbols, key=lambda s: (s.file, s.line))
            rendered = []
            for sym in symbols[:max_symbols]:
                signature = f" {sym.signature}" if sym.signature else ""
                rendered.append(f"{sym.file}:{sym.line}: {sym.kind} {sym.name}{signature}")
            header = f"Found {len(symbols)} symbol(s); showing {min(len(symbols), max_symbols)}."
            return ToolResult(success=True, content=self._truncate(header + "\n" + "\n".join(rendered), max_chars))
        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
        except Exception as e:
            return ToolResult(success=False, content=f"Error listing symbols: {type(e).__name__}: {e}")

    def _symbols_for_file(self, path: Path, include_private: bool) -> list[Symbol]:
        if path.suffix.lower() == ".py":
            return self._python_symbols(path, include_private)
        return self._regex_symbols(path, include_private)

    def _python_symbols(self, path: Path, include_private: bool) -> list[Symbol]:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return []

        symbols: list[Symbol] = []
        display = self._display_path(path)

        def visit(nodes: list[ast.stmt], prefix: str = "") -> None:
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = node.name
                    if not include_private and name.startswith("_"):
                        continue
                    kind = "class" if isinstance(node, ast.ClassDef) else "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    full_name = f"{prefix}.{name}" if prefix else name
                    symbols.append(Symbol(display, node.lineno, kind, full_name, self._python_signature(node)))
                    if isinstance(node, ast.ClassDef):
                        visit(node.body, full_name)

        visit(tree.body)
        return symbols

    def _python_signature(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            bases = [self._safe_unparse(base) for base in node.bases]
            return f"({', '.join(bases)})" if bases else ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"({', '.join(arg.arg for arg in node.args.args)})"
        return ""

    def _safe_unparse(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "?"

    def _regex_symbols(self, path: Path, include_private: bool) -> list[Symbol]:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

        patterns = [
            ("class", re.compile(r"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+|internal\s+|final\s+|abstract\s+|data\s+|sealed\s+|open\s+)*class\s+([A-Za-z_$][\w$]*)")),
            ("interface", re.compile(r"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+|internal\s+)*interface\s+([A-Za-z_$][\w$]*)")),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")),
            ("method", re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+|suspend\s+|override\s+|async\s+)*[A-Za-z_$][\w$<>\[\], ?]*\s+([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*(?:\{|=|throws)")),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>")),
            ("function", re.compile(r"^\s*func\s+([A-Za-z_$][\w$]*)\s*\(")),
            ("function", re.compile(r"^\s*fn\s+([A-Za-z_$][\w$]*)\s*\(")),
        ]
        display = self._display_path(path)
        symbols: list[Symbol] = []
        for idx, line in enumerate(lines, start=1):
            for kind, pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                name = match.group(1)
                if not include_private and name.startswith("_"):
                    continue
                symbols.append(Symbol(display, idx, kind, name, line.strip()))
                break
        return symbols


class FindRefsTool(CodeSearchMixin, BaseTool):
    name = "find_refs"
    description = (
        "Find references to a symbol with line numbers and small context windows. "
        "Use this instead of read_file when tracing call sites, usages, and impact before editing."
    )
    args_schema = FindRefsArgs

    async def execute(self, **kwargs: object) -> ToolResult:
        symbol = str(kwargs.get("symbol", "")).strip()
        search_path = str(kwargs.get("path", "."))
        include = kwargs.get("include")
        include_definitions = bool(kwargs.get("include_definitions", False))
        context_lines = int(kwargs.get("context_lines", 1))
        max_matches = int(kwargs.get("max_matches", 80))
        max_chars = int(kwargs.get("max_chars", 16000))

        if not symbol:
            return ToolResult(success=False, content="Error: symbol must not be empty")

        try:
            matcher = self._identifier_re(symbol)
            root = self._safe_root(search_path)
            matches: list[str] = []
            total = 0

            for path in self._iter_text_files(root, str(include) if include else None):
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                for idx, line in enumerate(lines, start=1):
                    if not matcher.search(line):
                        continue
                    if not include_definitions and self._looks_like_definition_line(line, symbol):
                        continue
                    total += 1
                    if len(matches) >= max_matches:
                        continue
                    matches.extend(
                        self._render_line_range(
                            path=path,
                            lines=lines,
                            center_line=idx,
                            context_lines=context_lines,
                        )
                    )
                    if context_lines:
                        matches.append("--")

            if not matches:
                return ToolResult(success=True, content=f"No references found for symbol: {symbol}")

            header = f"Found {total} reference(s) for {symbol!r}; showing {min(total, max_matches)}."
            return ToolResult(success=True, content=self._truncate(header + "\n" + "\n".join(matches), max_chars))
        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
        except re.error as e:
            return ToolResult(success=False, content=f"Invalid symbol pattern: {e}")
        except Exception as e:
            return ToolResult(success=False, content=f"Error finding references: {type(e).__name__}: {e}")

    def _looks_like_definition_line(self, line: str, symbol: str) -> bool:
        tail = re.escape(self._identifier_tail(symbol))
        definition_patterns = (
            rf"^\s*(?:async\s+)?def\s+{tail}\s*\(",
            rf"^\s*class\s+{tail}\b",
            rf"^\s*(?:export\s+)?(?:async\s+)?function\s+{tail}\s*\(",
            rf"^\s*(?:export\s+)?(?:const|let|var)\s+{tail}\s*=",
            rf"^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+|suspend\s+|override\s+)*[A-Za-z_$][\w$<>\[\], ?]*\s+{tail}\s*\([^;]*\)\s*(?:\{{|=|throws)",
            rf"^\s*func\s+{tail}\s*\(",
            rf"^\s*fn\s+{tail}\s*\(",
        )
        return any(re.search(pattern, line) for pattern in definition_patterns)


class GotoDefTool(CodeSearchMixin, BaseTool):
    name = "goto_def"
    description = (
        "Locate likely definitions of a symbol and return only the surrounding line range. "
        "Use this before read_lines when jumping to a class/function/method implementation."
    )
    args_schema = GotoDefArgs

    async def execute(self, **kwargs: object) -> ToolResult:
        symbol = str(kwargs.get("symbol", "")).strip()
        search_path = str(kwargs.get("path", "."))
        include = kwargs.get("include")
        context_lines = int(kwargs.get("context_lines", 4))
        max_matches = int(kwargs.get("max_matches", 20))
        max_chars = int(kwargs.get("max_chars", 16000))

        if not symbol:
            return ToolResult(success=False, content="Error: symbol must not be empty")

        try:
            root = self._safe_root(search_path)
            definitions: list[tuple[Path, int, str]] = []
            symbol_tail = self._identifier_tail(symbol)

            for path in self._iter_text_files(root, str(include) if include else None):
                if path.suffix.lower() not in SYMBOL_EXTENSIONS:
                    continue
                for found_symbol in self._symbols_for_file(path, include_private=True):
                    if self._symbol_matches(found_symbol.name, symbol, symbol_tail):
                        definitions.append((path, found_symbol.line, found_symbol.signature or found_symbol.kind))
                        if len(definitions) >= max_matches:
                            break
                if len(definitions) >= max_matches:
                    break

            if not definitions:
                return ToolResult(success=True, content=f"No definition found for symbol: {symbol}")

            rendered: list[str] = []
            for path, line_no, signature in definitions[:max_matches]:
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                rendered.append(f"Definition: {self._display_path(path)}:{line_no} {signature}".rstrip())
                rendered.extend(
                    self._render_line_range(
                        path=path,
                        lines=lines,
                        center_line=line_no,
                        context_lines=context_lines,
                    )
                )
                rendered.append("--")

            header = f"Found {len(definitions)} definition candidate(s) for {symbol!r}; showing {min(len(definitions), max_matches)}."
            return ToolResult(success=True, content=self._truncate(header + "\n" + "\n".join(rendered), max_chars))
        except PermissionError as e:
            return ToolResult(success=False, content=str(e))
        except Exception as e:
            return ToolResult(success=False, content=f"Error locating definition: {type(e).__name__}: {e}")

    def _symbol_matches(self, candidate: str, requested: str, requested_tail: str) -> bool:
        normalized = requested.strip()
        if not normalized:
            return False
        return candidate == normalized or candidate.split(".")[-1] == requested_tail

    def _symbols_for_file(self, path: Path, include_private: bool) -> list[Symbol]:
        if path.suffix.lower() == ".py":
            return self._python_symbols(path, include_private)
        return self._regex_symbols(path, include_private)

    def _python_symbols(self, path: Path, include_private: bool) -> list[Symbol]:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return []

        symbols: list[Symbol] = []
        display = self._display_path(path)

        def visit(nodes: list[ast.stmt], prefix: str = "") -> None:
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name = node.name
                    if not include_private and name.startswith("_"):
                        continue
                    kind = "class" if isinstance(node, ast.ClassDef) else "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    full_name = f"{prefix}.{name}" if prefix else name
                    symbols.append(Symbol(display, node.lineno, kind, full_name, self._python_signature(node)))
                    if isinstance(node, ast.ClassDef):
                        visit(node.body, full_name)

        visit(tree.body)
        return symbols

    def _python_signature(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            bases = [self._safe_unparse(base) for base in node.bases]
            return f"({', '.join(bases)})" if bases else ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return f"({', '.join(arg.arg for arg in node.args.args)})"
        return ""

    def _safe_unparse(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return "?"

    def _regex_symbols(self, path: Path, include_private: bool) -> list[Symbol]:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []

        patterns = [
            ("class", re.compile(r"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+|internal\s+|final\s+|abstract\s+|data\s+|sealed\s+|open\s+)*class\s+([A-Za-z_$][\w$]*)")),
            ("interface", re.compile(r"^\s*(?:export\s+)?(?:public\s+|private\s+|protected\s+|internal\s+)*interface\s+([A-Za-z_$][\w$]*)")),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(")),
            ("method", re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+|suspend\s+|override\s+|async\s+)*[A-Za-z_$][\w$<>\[\], ?]*\s+([A-Za-z_$][\w$]*)\s*\([^;]*\)\s*(?:\{|=|throws)")),
            ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>")),
            ("function", re.compile(r"^\s*func\s+([A-Za-z_$][\w$]*)\s*\(")),
            ("function", re.compile(r"^\s*fn\s+([A-Za-z_$][\w$]*)\s*\(")),
        ]
        display = self._display_path(path)
        symbols: list[Symbol] = []
        for idx, line in enumerate(lines, start=1):
            for kind, pattern in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                name = match.group(1)
                if not include_private and name.startswith("_"):
                    continue
                symbols.append(Symbol(display, idx, kind, name, line.strip()))
                break
        return symbols
