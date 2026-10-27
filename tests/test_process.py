"""Tests for perf_agent/infra/process.py (spec 04.1)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from perf_agent.infra.process import (
    COMMAND_NOT_FOUND,
    MAX_OUTPUT_CHARS,
    _resolve_executable,
    run,
)

PY = sys.executable


def test_timeout_kills_the_command_quickly(tmp_path: Path) -> None:
    start = time.perf_counter()

    result = run([PY, "-c", "import time; time.sleep(5)"], tmp_path, timeout_s=0.2)

    assert result.timed_out is True
    assert result.ok is False
    assert time.perf_counter() - start < 1


def test_captures_output_and_exit_code(tmp_path: Path) -> None:
    code = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"

    result = run([PY, "-c", code], tmp_path, timeout_s=10)

    assert (result.returncode, result.stdout.strip(), result.stderr.strip()) == (3, "out", "err")
    assert result.timed_out is False
    assert result.ok is False
    assert result.duration_ns > 0


def test_success(tmp_path: Path) -> None:
    assert run([PY, "-c", "pass"], tmp_path, timeout_s=10).ok is True


def test_runs_in_the_given_folder(tmp_path: Path) -> None:
    result = run([PY, "-c", "import os; print(os.getcwd())"], tmp_path, timeout_s=10)

    assert Path(result.stdout.strip()).resolve() == tmp_path.resolve()


def test_secrets_do_not_reach_the_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("NEO4J_PASSWORD", "hunter2")
    code = "import os; print(os.environ.get('ANTHROPIC_API_KEY'), os.environ.get('NEO4J_PASSWORD'))"

    result = run([PY, "-c", code], tmp_path, timeout_s=10)

    assert result.stdout.strip() == "None None"


def test_extra_env_is_passed(tmp_path: Path) -> None:
    code = "import os; print(os.environ['PERF_MODE'])"

    result = run([PY, "-c", code], tmp_path, timeout_s=10, env={"PERF_MODE": "fast"})

    assert result.stdout.strip() == "fast"


def test_missing_program_is_reported_not_raised(tmp_path: Path) -> None:
    result = run(["no-such-program-perf-agent"], tmp_path, timeout_s=10)

    assert result.returncode == COMMAND_NOT_FOUND
    assert "cannot start 'no-such-program-perf-agent'" in result.stderr


def test_long_output_keeps_its_end(tmp_path: Path) -> None:
    code = f"print('a' * {MAX_OUTPUT_CHARS} + 'END')"

    result = run([PY, "-c", code], tmp_path, timeout_s=10)

    assert result.stdout.startswith("[... ")
    assert result.stdout.rstrip().endswith("END")
    assert len(result.stdout) < MAX_OUTPUT_CHARS + 100


def test_output_can_be_kept_whole(tmp_path: Path) -> None:
    code = f"print('a' * {MAX_OUTPUT_CHARS * 2})"

    result = run([PY, "-c", code], tmp_path, timeout_s=10, max_output_chars=None)

    assert len(result.stdout.strip()) == MAX_OUTPUT_CHARS * 2


def test_relative_program_path_is_resolved_against_cwd(tmp_path: Path) -> None:
    assert _resolve_executable(["./build.sh", "x"], tmp_path) == [
        str((tmp_path / "build.sh").resolve()),
        "x",
    ]
    assert _resolve_executable(["java", "-version"], tmp_path) == ["java", "-version"]
    resolved = _resolve_executable(["bin/run", "x"], tmp_path)
    assert resolved[0] == str((tmp_path / "bin/run").resolve())


def test_empty_command_is_a_programming_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty command"):
        run([], tmp_path, timeout_s=1)
