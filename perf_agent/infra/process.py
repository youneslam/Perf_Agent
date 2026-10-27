"""The single entry point for running external commands (spec 04.1).

Every subprocess of the agent goes through `run`: no shell, a timeout that kills the whole
process tree, captured output and a minimal environment, so secrets such as API keys never
reach the (LLM-written) code being built, tested and benchmarked.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 200_000
COMMAND_NOT_FOUND = 127

_ENV_KEYS = ("PATH", "JAVA_HOME", "HOME", "LANG")
# Without these, Windows programs (including java.exe and python.exe) fail to start.
_WINDOWS_ENV_KEYS = (
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
)


class ProcessResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ns: int
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def minimal_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The few variables a build or a JVM needs, plus `extra`. Nothing else is inherited."""
    keys = _ENV_KEYS + (_WINDOWS_ENV_KEYS if sys.platform == "win32" else ())
    env = {key: os.environ[key] for key in keys if key in os.environ}
    env.update(extra or {})
    return env


def run(
    cmd: list[str],
    cwd: Path,
    timeout_s: float,
    env: dict[str, str] | None = None,
    max_output_chars: int | None = MAX_OUTPUT_CHARS,
) -> ProcessResult:
    """Run `cmd` in `cwd` and wait at most `timeout_s` seconds.

    On timeout the process and its children are killed and `timed_out=True` is returned;
    this function never raises for a timeout or a failing command. `env` adds variables to
    the minimal environment. Output longer than `max_output_chars` keeps only its end
    (`None` keeps everything, e.g. when the output must be hashed).
    """
    argv = _resolve_executable(cmd, cwd)
    logger.debug("running %s in %s (timeout %ss)", argv, cwd, timeout_s)
    start = time.perf_counter_ns()
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=minimal_env(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_creation_flags(),
            start_new_session=sys.platform != "win32",
        )
    except OSError as exc:
        return ProcessResult(
            cmd=cmd,
            returncode=COMMAND_NOT_FOUND,
            stdout="",
            stderr=f"cannot start {cmd[0]!r}: {exc}",
            duration_ns=time.perf_counter_ns() - start,
            timed_out=False,
        )

    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        logger.warning("command timed out after %ss: %s", timeout_s, cmd)
    duration_ns = time.perf_counter_ns() - start

    return ProcessResult(
        cmd=cmd,
        returncode=process.returncode if process.returncode is not None else -1,
        stdout=_truncate(stdout or "", max_output_chars),
        stderr=_truncate(stderr or "", max_output_chars),
        duration_ns=duration_ns,
        timed_out=timed_out,
    )


def _resolve_executable(cmd: list[str], cwd: Path) -> list[str]:
    """Make a relative program path such as `./build.sh` relative to `cwd` on every OS."""
    if not cmd:
        raise ValueError("empty command")
    program = Path(cmd[0])
    has_separator = "/" in cmd[0] or os.sep in cmd[0]
    if has_separator and not program.is_absolute():
        return [str((cwd / program).resolve()), *cmd[1:]]
    return list(cmd)


def _creation_flags() -> int:
    """On Windows, start the command in its own process group so it can be killed as a tree."""
    if sys.platform == "win32":
        return subprocess.CREATE_NEW_PROCESS_GROUP
    return 0


def _kill_tree(process: subprocess.Popen[str]) -> None:
    """Kill the process and everything it started."""
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        pass
    process.kill()


def _truncate(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"[... {dropped} characters truncated ...]\n{text[-limit:]}"
