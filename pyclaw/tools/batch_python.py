from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from .base import BaseTool, ToolResult


class BatchPythonArgs(BaseModel):
    task_name: str = Field(
        min_length=1,
        description="Short name for this batch job. Used only to create a durable job directory.",
    )
    targets: list[str] = Field(
        description="Batch targets, one item per line in the generated targets.txt file."
    )
    script: str = Field(
        min_length=1,
        description=(
            "Python script to run once. It runs with cwd set to the job directory, "
            "should read targets.txt, and must write summary.json. Prefer also writing "
            "results.jsonl or results.csv for item-level evidence."
        ),
    )
    timeout: int = Field(default=300, ge=1, le=3600, description="Timeout in seconds")
    expected_outputs: list[str] = Field(
        default_factory=lambda: ["summary.json"],
        description=(
            "Relative output paths expected inside the job directory. The tool always "
            "requires summary.json for a successful run."
        ),
    )


class BatchPythonTool(BaseTool):
    """Run one durable Python batch job and return file-backed evidence."""

    name = "batch_python"
    description = (
        "Run a Python batch script once in a durable workspace job directory. "
        "Use this for multi-item operational work such as querying many pods, devices, "
        "services, URLs, or records. The script runs with cwd=<job_dir>, reads the "
        "generated targets.txt, and writes summary.json plus item-level result files. "
        "The tool returns paths to summary/log/results so the model can summarize files "
        "instead of calling tools once per item."
    )
    args_schema = BatchPythonArgs

    JOBS_DIR = ".pyclaw_batch_jobs"
    SUMMARY_FILE = "summary.json"
    RESULTS_JSONL_FILE = "results.jsonl"
    RESULTS_CSV_FILE = "results.csv"
    TARGETS_FILE = "targets.txt"
    SCRIPT_FILE = "batch_task.py"
    LOG_FILE = "run.log"
    SITECUSTOMIZE_FILE = "sitecustomize.py"

    async def execute(self, **kwargs: object) -> ToolResult:
        task_name = str(kwargs.get("task_name", "")).strip()
        targets = [str(item) for item in (kwargs.get("targets") or [])]
        script = str(kwargs.get("script", ""))
        timeout = int(kwargs.get("timeout", 300))
        expected_outputs = [str(item) for item in (kwargs.get("expected_outputs") or [self.SUMMARY_FILE])]
        if self.SUMMARY_FILE not in expected_outputs:
            expected_outputs.insert(0, self.SUMMARY_FILE)

        proc: asyncio.subprocess.Process | None = None
        stdout = ""
        stderr = ""
        exit_code = -1
        timed_out = False
        job_dir: Path | None = None
        log_path: Path | None = None
        expected_paths: dict[str, Path] = {}

        try:
            job_dir = await asyncio.to_thread(self._create_job_dir, task_name)
            targets_path = self._path_in_job(job_dir, self.TARGETS_FILE)
            script_path = self._path_in_job(job_dir, self.SCRIPT_FILE)
            log_path = self._path_in_job(job_dir, self.LOG_FILE)
            sitecustomize_path = self._path_in_job(job_dir, self.SITECUSTOMIZE_FILE)
            summary_path = self._path_in_job(job_dir, self.SUMMARY_FILE)
            results_jsonl_path = self._path_in_job(job_dir, self.RESULTS_JSONL_FILE)
            results_csv_path = self._path_in_job(job_dir, self.RESULTS_CSV_FILE)
            expected_paths = {
                relative_path: self._path_in_job(job_dir, relative_path)
                for relative_path in self._dedupe(expected_outputs)
            }

            await asyncio.gather(
                asyncio.to_thread(targets_path.write_text, "\n".join(targets) + ("\n" if targets else ""), encoding="utf-8"),
                asyncio.to_thread(script_path.write_text, script, encoding="utf-8"),
                asyncio.to_thread(sitecustomize_path.write_text, self._sitecustomize_source(), encoding="utf-8"),
            )

            env = self._subprocess_env(job_dir=job_dir, targets_path=targets_path)
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(job_dir),
                env=env,
                start_new_session=True,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                timed_out = True
                await self._terminate_process(proc)
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=2)
                except Exception:
                    stdout_bytes, stderr_bytes = b"", b""

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = int(proc.returncode if proc.returncode is not None else -1)

            await asyncio.to_thread(
                log_path.write_text,
                self._render_run_log(
                    job_dir=job_dir,
                    script_path=script_path,
                    timeout=timeout,
                    timed_out=timed_out,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                ),
                encoding="utf-8",
            )

            missing_outputs = await asyncio.to_thread(
                self._missing_outputs,
                expected_paths,
                job_dir,
            )
            summary_exists = self._safe_exists(summary_path, job_dir)
            summary: dict[str, Any] = {}
            parse_error = ""
            if summary_exists:
                try:
                    summary = await asyncio.to_thread(self._read_summary_json, summary_path, job_dir)
                except (json.JSONDecodeError, ValueError) as exc:
                    parse_error = str(exc)

            counts = self._summary_counts(summary, target_count=len(targets))
            structured = self._structured_payload(
                job_dir=job_dir,
                targets_path=targets_path,
                script_path=script_path,
                log_path=log_path,
                summary_path=summary_path,
                results_jsonl_path=results_jsonl_path if await asyncio.to_thread(self._safe_exists, results_jsonl_path, job_dir) else None,
                results_csv_path=results_csv_path if await asyncio.to_thread(self._safe_exists, results_csv_path, job_dir) else None,
                target_count=len(targets),
                counts=counts,
                exit_code=exit_code,
                timed_out=timed_out,
                expected_outputs=list(expected_paths.keys()),
                missing_outputs=missing_outputs,
                stdout=stdout,
                stderr=stderr,
                summary=summary,
            )

            if timed_out:
                return ToolResult(
                    success=False,
                    content=self._render_content(
                        status="timed out",
                        structured=structured,
                        stderr=stderr,
                    ),
                    metadata=self._metadata(structured),
                    structured=structured,
                    error_code="timeout",
                    retryable=True,
                )

            if exit_code == 0 and not summary_exists:
                return ToolResult(
                    success=False,
                    content=self._render_content(
                        status="missing summary.json",
                        structured=structured,
                        stderr=stderr,
                    ),
                    metadata=self._metadata(structured),
                    structured=structured,
                    error_code="missing_summary",
                    requires_model_repair=True,
                )

            if exit_code == 0 and parse_error:
                structured["summary_parse_error"] = parse_error
                return ToolResult(
                    success=False,
                    content=self._render_content(
                        status="invalid summary.json",
                        structured=structured,
                        stderr=parse_error,
                    ),
                    metadata=self._metadata(structured),
                    structured=structured,
                    error_code="invalid_summary",
                    requires_model_repair=True,
                )

            if exit_code != 0:
                return ToolResult(
                    success=False,
                    content=self._render_content(
                        status="failed",
                        structured=structured,
                        stderr=stderr,
                    ),
                    metadata=self._metadata(structured),
                    structured=structured,
                    error_code="nonzero_exit",
                    retryable=True,
                    requires_model_repair=True,
                )

            if missing_outputs:
                return ToolResult(
                    success=False,
                    content=self._render_content(
                        status="missing expected outputs",
                        structured=structured,
                        stderr="",
                    ),
                    metadata=self._metadata(structured),
                    structured=structured,
                    error_code="missing_outputs",
                    requires_model_repair=True,
                )

            return ToolResult(
                success=True,
                content=self._render_content(status="completed", structured=structured, stderr=""),
                metadata=self._metadata(structured),
                structured=structured,
            )

        except PermissionError as exc:
            return ToolResult(
                success=False,
                content=f"Batch Python sandbox denied execution: {exc}",
                metadata={"job_dir": str(job_dir) if job_dir else ""},
                structured={
                    "operation": "batch_python",
                    "job_dir": str(job_dir) if job_dir else "",
                    "log_path": str(log_path) if log_path else "",
                },
                error_code="sandbox_denied",
                requires_model_repair=True,
            )
        except ValueError as exc:
            return ToolResult(
                success=False,
                content=f"Invalid batch_python arguments: {exc}",
                metadata={"job_dir": str(job_dir) if job_dir else ""},
                structured={
                    "operation": "batch_python",
                    "job_dir": str(job_dir) if job_dir else "",
                    "log_path": str(log_path) if log_path else "",
                },
                error_code="invalid_arguments",
                requires_model_repair=True,
            )
        except asyncio.CancelledError:
            if proc is not None:
                await self._terminate_process(proc)
            raise
        except Exception as exc:
            if log_path is not None:
                try:
                    await asyncio.to_thread(
                        log_path.write_text,
                        self._render_run_log(
                            job_dir=job_dir or Path("."),
                            script_path=Path(""),
                            timeout=timeout,
                            timed_out=timed_out,
                            exit_code=exit_code,
                            stdout=stdout,
                            stderr=f"{type(exc).__name__}: {exc}\n{stderr}",
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            return ToolResult(
                success=False,
                content=f"Error executing batch Python job: {type(exc).__name__}: {exc}",
                metadata={"job_dir": str(job_dir) if job_dir else ""},
                structured={
                    "operation": "batch_python",
                    "job_dir": str(job_dir) if job_dir else "",
                    "log_path": str(log_path) if log_path else "",
                    "exception": type(exc).__name__,
                },
                error_code="execution_error",
                requires_model_repair=True,
            )

    def _create_job_dir(self, task_name: str) -> Path:
        root = self._workspace_root()
        jobs_dir = Path(self.validate_path(str(root / self.JOBS_DIR)))
        jobs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._safe_task_name(task_name)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        unique = uuid.uuid4().hex[:8]
        job_dir = Path(self.validate_path(str(jobs_dir / f"{safe_name}-{timestamp}-{unique}")))
        job_dir.mkdir(parents=False, exist_ok=False)
        self._ensure_under(job_dir, jobs_dir)
        return job_dir

    def _workspace_root(self) -> Path:
        root = Path(self.work_dir or os.getcwd()).expanduser()
        return Path(self.validate_path(str(root)))

    def _path_in_job(self, job_dir: Path, relative_path: str) -> Path:
        relative = str(relative_path or "").strip()
        if not relative:
            raise ValueError("expected output path must not be empty")
        if os.path.isabs(relative):
            raise ValueError(f"expected output path must be relative: {relative}")
        parts = Path(relative).parts
        if any(part in {"..", ""} for part in parts):
            raise ValueError(f"expected output path must stay inside the job directory: {relative}")
        path = Path(self.validate_path(str(job_dir / relative)))
        self._ensure_under(path, job_dir)
        return path

    def _ensure_under(self, path: Path, root: Path) -> None:
        path_real = os.path.realpath(os.path.abspath(str(path)))
        root_real = os.path.realpath(os.path.abspath(str(root)))
        if os.path.commonpath([path_real, root_real]) != root_real:
            raise PermissionError(f"Path '{path}' is outside job directory '{root}'")

    def _safe_exists(self, path: Path, job_dir: Path) -> bool:
        try:
            validated = Path(self.validate_path(str(path)))
            self._ensure_under(validated, job_dir)
            return validated.exists()
        except PermissionError:
            return False

    def _missing_outputs(self, expected_paths: dict[str, Path], job_dir: Path) -> list[str]:
        return [relative for relative, path in expected_paths.items() if not self._safe_exists(path, job_dir)]

    def _read_summary_json(self, summary_path: Path, job_dir: Path) -> dict[str, Any]:
        safe_path = Path(self.validate_path(str(summary_path)))
        self._ensure_under(safe_path, job_dir)
        with safe_path.open("r", encoding="utf-8", errors="replace") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, dict):
            raise ValueError("summary.json must contain a JSON object")
        return parsed

    def _subprocess_env(self, *, job_dir: Path, targets_path: Path) -> dict[str, str]:
        env = dict(os.environ)
        env["PYCLAW_BATCH_JOB_DIR"] = str(job_dir)
        env["PYCLAW_BATCH_TARGETS"] = str(targets_path)
        env["PYCLAW_WORK_DIR"] = str(self._workspace_root())
        allowed_roots = [str(self._workspace_root())]
        allowed_roots.extend(str(Path(path).expanduser()) for path in self.allowed_paths)
        env["PYCLAW_ALLOWED_PATHS"] = os.pathsep.join(allowed_roots)
        env["PYTHONUNBUFFERED"] = "1"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (str(job_dir), existing_pythonpath) if path
        )
        return env

    def _render_run_log(
        self,
        *,
        job_dir: Path,
        script_path: Path,
        timeout: int,
        timed_out: bool,
        exit_code: int,
        stdout: str,
        stderr: str,
    ) -> str:
        lines = [
            "Batch Python run log",
            f"cwd: {job_dir}",
            f"script: {script_path}",
            f"python: {sys.executable}",
            f"timeout_seconds: {timeout}",
            f"timed_out: {str(timed_out).lower()}",
            f"exit_code: {exit_code}",
            "",
            "STDOUT:",
            stdout,
            "",
            "STDERR:",
            stderr,
        ]
        if timed_out:
            lines.insert(7, f"Timed out after {timeout} seconds")
        return "\n".join(lines)

    def _structured_payload(
        self,
        *,
        job_dir: Path,
        targets_path: Path,
        script_path: Path,
        log_path: Path,
        summary_path: Path,
        results_jsonl_path: Path | None,
        results_csv_path: Path | None,
        target_count: int,
        counts: dict[str, int],
        exit_code: int,
        timed_out: bool,
        expected_outputs: list[str],
        missing_outputs: list[str],
        stdout: str,
        stderr: str,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "operation": "batch_python",
            "job_dir": str(job_dir),
            "targets_path": str(targets_path),
            "script_path": str(script_path),
            "log_path": str(log_path),
            "summary_path": str(summary_path),
            "results_jsonl_path": str(results_jsonl_path) if results_jsonl_path else "",
            "results_csv_path": str(results_csv_path) if results_csv_path else "",
            "target_count": target_count,
            "total": counts["total"],
            "success_count": counts["success_count"],
            "failed_count": counts["failed_count"],
            "missing_count": counts["missing_count"],
            "exit_code": exit_code,
            "timed_out": timed_out,
            "expected_outputs": expected_outputs,
            "missing_outputs": missing_outputs,
            "stdout_tail": self._tail(stdout),
            "stderr_tail": self._tail(stderr),
            "summary": summary,
        }

    def _metadata(self, structured: dict[str, Any]) -> dict[str, Any]:
        return {
            "operation": "batch_python",
            "job_dir": structured.get("job_dir", ""),
            "log_path": structured.get("log_path", ""),
            "summary_path": structured.get("summary_path", ""),
            "total": structured.get("total", 0),
            "success_count": structured.get("success_count", 0),
            "failed_count": structured.get("failed_count", 0),
            "missing_count": structured.get("missing_count", 0),
            "exit_code": structured.get("exit_code", -1),
            "timed_out": structured.get("timed_out", False),
        }

    def _render_content(self, *, status: str, structured: dict[str, Any], stderr: str) -> str:
        total = int(structured.get("total") or 0)
        success = int(structured.get("success_count") or 0)
        failed = int(structured.get("failed_count") or 0)
        missing = int(structured.get("missing_count") or 0)
        lines = [
            f"Batch Python job {status}",
            f"total={total} success={success} failed={failed} missing={missing}",
            f"Job directory: {structured.get('job_dir', '')}",
            f"Targets file: {structured.get('targets_path', '')}",
            f"Script file: {structured.get('script_path', '')}",
            f"Summary file: {structured.get('summary_path', '')}",
        ]
        if structured.get("results_jsonl_path"):
            lines.append(f"Result JSONL file: {structured['results_jsonl_path']}")
        if structured.get("results_csv_path"):
            lines.append(f"Result CSV file: {structured['results_csv_path']}")
        lines.append(f"Log file: {structured.get('log_path', '')}")
        if structured.get("missing_outputs"):
            lines.append(f"Missing expected outputs: {', '.join(structured['missing_outputs'])}")
        if stderr:
            lines.extend(["", "Error tail:", self._tail(stderr, max_chars=2000)])
        return "\n".join(lines)

    def _summary_counts(self, summary: dict[str, Any], *, target_count: int) -> dict[str, int]:
        total = self._int_from_mapping(
            summary,
            ("total", "target_count", "targets", "count", "items", "processed", "total_count"),
        )
        success = self._int_from_mapping(
            summary,
            ("success_count", "succeeded", "successful", "success", "ok", "passed", "completed"),
        )
        failed = self._int_from_mapping(
            summary,
            ("failed_count", "failure_count", "failures", "failed", "errors", "error_count"),
        )
        missing = self._int_from_mapping(summary, ("missing_count", "missing", "skipped", "not_found"))

        if total < 0:
            total = target_count
        if success < 0 and total >= 0 and failed >= 0 and missing >= 0:
            success = max(0, total - failed - missing)
        elif success < 0:
            success = 0
        if failed < 0 and total >= 0 and success >= 0 and missing >= 0:
            failed = max(0, total - success - missing)
        elif failed < 0:
            failed = 0
        if missing < 0 and total >= 0 and success >= 0 and failed >= 0:
            missing = max(0, total - success - failed)
        elif missing < 0:
            missing = 0
        return {
            "total": max(0, int(total)),
            "success_count": max(0, int(success)),
            "failed_count": max(0, int(failed)),
            "missing_count": max(0, int(missing)),
        }

    def _int_from_mapping(self, mapping: dict[str, Any], keys: Iterable[str]) -> int:
        lowered = {str(key).strip().lower(): value for key, value in mapping.items()}
        for key in keys:
            value = lowered.get(key.lower())
            parsed = self._coerce_int(value)
            if parsed >= 0:
                return parsed
        return -1

    def _coerce_int(self, value: Any) -> int:
        if isinstance(value, bool):
            return -1
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, (list, tuple, set, dict)):
            return len(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+", value)
            if match:
                return int(match.group(0))
        return -1

    def _safe_task_name(self, task_name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", task_name.strip().lower()).strip("-._")
        if not normalized:
            normalized = "batch"
        return normalized[:64]

    def _dedupe(self, items: Iterable[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _tail(self, text: str, *, max_chars: int = 4000) -> str:
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    async def _terminate_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=2)
        except Exception:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                await proc.wait()
            except Exception:
                pass

    def _sitecustomize_source(self) -> str:
        return r'''
from __future__ import annotations

import builtins
import io
import os
import pathlib
import sys

_WORK_DIR = os.path.realpath(os.path.abspath(os.environ.get("PYCLAW_WORK_DIR") or os.getcwd()))
_ALLOWED = []
for _raw in os.environ.get("PYCLAW_ALLOWED_PATHS", "").split(os.pathsep):
    if _raw:
        _ALLOWED.append(os.path.realpath(os.path.abspath(os.path.expanduser(_raw))))
if _WORK_DIR not in _ALLOWED:
    _ALLOWED.insert(0, _WORK_DIR)
_SYSTEM_READ_ROOTS = tuple(
    os.path.realpath(os.path.abspath(root))
    for root in {
        sys.prefix,
        getattr(sys, "base_prefix", sys.prefix),
        getattr(sys, "exec_prefix", sys.prefix),
        os.path.dirname(os.__file__),
        os.path.dirname(os.path.dirname(os.__file__)),
    }
    if root
)


def _real_path(path):
    if path is None or isinstance(path, int):
        return None
    text = os.fspath(path)
    if not os.path.isabs(text):
        text = os.path.join(os.getcwd(), text)
    return os.path.realpath(os.path.abspath(os.path.expanduser(text)))


def _under(path, roots):
    if path is None:
        return True
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:
            pass
    return False


def _is_write_mode(mode):
    text = str(mode or "r")
    return any(flag in text for flag in ("w", "a", "x", "+"))


def _check(path, mode="r"):
    real = _real_path(path)
    if real is None:
        return
    if _under(real, _ALLOWED):
        return
    if not _is_write_mode(mode) and _under(real, _SYSTEM_READ_ROOTS):
        return
    raise PermissionError(f"Batch Python sandbox denied file access outside workspace: {path}")


_original_open = builtins.open
_original_io_open = io.open
_original_os_open = os.open
_original_unlink = os.unlink
_original_remove = os.remove
_original_rename = os.rename
_original_replace = os.replace
_original_mkdir = os.mkdir
_original_makedirs = os.makedirs


def _open(file, mode="r", *args, **kwargs):
    _check(file, mode)
    return _original_open(file, mode, *args, **kwargs)


def _io_open(file, mode="r", *args, **kwargs):
    _check(file, mode)
    return _original_io_open(file, mode, *args, **kwargs)


def _os_open(path, flags, mode=0o777, *args, **kwargs):
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    _check(path, "w" if flags & write_flags else "r")
    return _original_os_open(path, flags, mode, *args, **kwargs)


def _unlink(path, *args, **kwargs):
    _check(path, "w")
    return _original_unlink(path, *args, **kwargs)


def _remove(path, *args, **kwargs):
    _check(path, "w")
    return _original_remove(path, *args, **kwargs)


def _rename(src, dst, *args, **kwargs):
    _check(src, "w")
    _check(dst, "w")
    return _original_rename(src, dst, *args, **kwargs)


def _replace(src, dst, *args, **kwargs):
    _check(src, "w")
    _check(dst, "w")
    return _original_replace(src, dst, *args, **kwargs)


def _mkdir(path, mode=0o777, *args, **kwargs):
    _check(path, "w")
    return _original_mkdir(path, mode, *args, **kwargs)


def _makedirs(name, mode=0o777, exist_ok=False):
    _check(name, "w")
    return _original_makedirs(name, mode, exist_ok=exist_ok)


builtins.open = _open
io.open = _io_open
os.open = _os_open
os.unlink = _unlink
os.remove = _remove
os.rename = _rename
os.replace = _replace
os.mkdir = _mkdir
os.makedirs = _makedirs

_original_path_open = pathlib.Path.open
_original_path_read_text = pathlib.Path.read_text
_original_path_write_text = pathlib.Path.write_text
_original_path_read_bytes = pathlib.Path.read_bytes
_original_path_write_bytes = pathlib.Path.write_bytes
_original_path_unlink = pathlib.Path.unlink
_original_path_mkdir = pathlib.Path.mkdir
_original_path_rename = pathlib.Path.rename
_original_path_replace = pathlib.Path.replace


def _path_open(self, mode="r", *args, **kwargs):
    _check(self, mode)
    return _original_path_open(self, mode, *args, **kwargs)


def _path_read_text(self, *args, **kwargs):
    _check(self, "r")
    return _original_path_read_text(self, *args, **kwargs)


def _path_write_text(self, *args, **kwargs):
    _check(self, "w")
    return _original_path_write_text(self, *args, **kwargs)


def _path_read_bytes(self, *args, **kwargs):
    _check(self, "r")
    return _original_path_read_bytes(self, *args, **kwargs)


def _path_write_bytes(self, *args, **kwargs):
    _check(self, "w")
    return _original_path_write_bytes(self, *args, **kwargs)


def _path_unlink(self, *args, **kwargs):
    _check(self, "w")
    return _original_path_unlink(self, *args, **kwargs)


def _path_mkdir(self, *args, **kwargs):
    _check(self, "w")
    return _original_path_mkdir(self, *args, **kwargs)


def _path_rename(self, target, *args, **kwargs):
    _check(self, "w")
    _check(target, "w")
    return _original_path_rename(self, target, *args, **kwargs)


def _path_replace(self, target, *args, **kwargs):
    _check(self, "w")
    _check(target, "w")
    return _original_path_replace(self, target, *args, **kwargs)


pathlib.Path.open = _path_open
pathlib.Path.read_text = _path_read_text
pathlib.Path.write_text = _path_write_text
pathlib.Path.read_bytes = _path_read_bytes
pathlib.Path.write_bytes = _path_write_bytes
pathlib.Path.unlink = _path_unlink
pathlib.Path.mkdir = _path_mkdir
pathlib.Path.rename = _path_rename
pathlib.Path.replace = _path_replace
'''.lstrip()
