"""Command-based Java project adapter (spec 04.2).

The adapter never knows how the Java project builds or tests itself: it only runs the
scenario's `build_cmd`, `test_cmd` and `run_cmd` inside a version folder.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from perf_agent.config.scenario import Scenario
from perf_agent.infra.process import ProcessResult, run
from perf_agent.models import BenchmarkResult, BuildResult, Combo, Measurement, TestResult

logger = logging.getLogger(__name__)

# Line prefix the Java runner prints with its own measured time, e.g. `#TIME_NS 123456`.
TIME_MARKER = "#TIME_NS"
# How much of a failed run's output is kept in a BenchmarkError message.
LOG_TAIL_CHARS = 2_000

# Regexes used to check the package and public class name of the new source.
_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
_PUBLIC_TYPE = re.compile(
    r"\bpublic\s+(?:(?:abstract|final|sealed|non-sealed|strictfp|static)\s+)*"
    r"(?:class|interface|enum|record)\s+(\w+)"
)


# Raised when the LLM's code renames the class or changes its package.
class ComponentContractError(Exception):
    """The new source changes the package or the public class name of the component."""


# Raised when code would be written into a real version instead of a trial.
class ComponentWriteError(Exception):
    """Refused to write the component outside a `*_trial` folder."""


# Raised when a benchmark run crashes, times out or prints an invalid time.
class BenchmarkError(Exception):
    """A benchmark run failed, timed out or printed an invalid `#TIME_NS` line."""


# Works on the Java project inside a version folder using the scenario's commands.
class JavaAdapter:
    # Keep the component and commands, and split class_name into package + class name.
    def __init__(self, scenario: Scenario) -> None:
        self._component = scenario.component
        self._project = scenario.project
        package, _, simple_name = scenario.component.class_name.rpartition(".")
        self._package = package
        self._class_name = simple_name

    # Full path of the component's .java file inside a version folder.
    def component_path(self, version_dir: Path) -> Path:
        return version_dir / self._component.source_file

    # Read the Java source of the component (sent to the Planner and Coder).
    def read_component(self, version_dir: Path) -> str:
        return self.component_path(version_dir).read_text(encoding="utf-8")

    # Write the LLM's new code, only into a *_trial folder and with the same class.
    def write_component(self, version_dir: Path, source: str) -> None:
        """Overwrite the component in a `*_trial` folder (same path, package and class name)."""
        if not version_dir.name.endswith("_trial"):
            raise ComponentWriteError(
                f"refusing to write the component into {version_dir}: not a *_trial folder"
            )
        self._check_contract(source)
        self.component_path(version_dir).write_text(source, encoding="utf-8", newline="")
        logger.debug("wrote component into %s", version_dir.name)

    # Run build_cmd in the folder; the log is sent back to the Coder if it fails.
    def build(self, version_dir: Path) -> BuildResult:
        result = run(self._project.build_cmd, version_dir, self._project.build_timeout_s)
        return BuildResult(ok=result.ok, log=_log(result), duration_ns=result.duration_ns)

    # Run test_cmd in the folder; failing tests reject the trial.
    def test(self, version_dir: Path) -> TestResult:
        result = run(self._project.test_cmd, version_dir, self._project.test_timeout_s)
        return TestResult(ok=result.ok, log=_log(result))

    # Run run_cmd once for one graph/query pair.
    def run_once(self, version_dir: Path, combo: Combo) -> ProcessResult:
        """Run `run_cmd` once for `combo`; the full output is kept so it can be hashed."""
        cmd = [
            arg.replace("{graph}", combo.graph).replace("{query}", combo.query)
            for arg in self._project.run_cmd
        ]
        return run(cmd, version_dir, self._project.run_timeout_s, max_output_chars=None)

    # Run every combo once and collect its time and output fingerprint.
    def benchmark(self, version_dir: Path, combos: list[Combo]) -> BenchmarkResult:
        """One execution per combo; warmup is the Java class's job."""
        measurements = []
        for combo in combos:
            result = self.run_once(version_dir, combo)
            if not result.ok:
                log_tail = _log(result)[-LOG_TAIL_CHARS:]
                raise BenchmarkError(f"{combo.key} failed in {version_dir.name}:\n{log_tail}")
            measurements.append(
                Measurement(
                    combo=combo,
                    time_ns=_time_ns(result, combo),
                    output_hash=output_hash(result.stdout),
                )
            )
            logger.debug("%s %s: %d ns", version_dir.name, combo.key, measurements[-1].time_ns)
        return BenchmarkResult(version=version_dir.name, measurements=measurements)

    # Check the new source keeps the expected package and public class name.
    def _check_contract(self, source: str) -> None:
        code = _COMMENTS.sub("", source)
        package_match = _PACKAGE.search(code)
        package = package_match.group(1) if package_match else ""
        if package != self._package:
            raise ComponentContractError(
                f"the new source declares package {package or '(none)'!r}, "
                f"expected {self._package or '(none)'!r}"
            )
        public_types = _PUBLIC_TYPE.findall(code)
        if self._class_name not in public_types:
            found = ", ".join(public_types) or "none"
            raise ComponentContractError(
                f"the new source must declare public class {self._class_name} "
                f"(public types found: {found}); never rename the class"
            )


# Fingerprint of the program's result, ignoring time lines and trailing spaces.
def output_hash(stdout: str) -> str:
    """sha256 of the program output, with trailing spaces and `#TIME…` lines removed."""
    lines = [line.rstrip() for line in stdout.splitlines() if not line.startswith("#TIME")]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# Time of one run: the printed #TIME_NS if present, else the process duration.
def _time_ns(result: ProcessResult, combo: Combo) -> int:
    """The last `#TIME_NS <n>` printed by the program, else the process wall-clock time."""
    lines = [line for line in result.stdout.splitlines() if line.startswith(TIME_MARKER)]
    if not lines:
        return result.duration_ns
    value = lines[-1].removeprefix(TIME_MARKER).strip()
    try:
        time_ns = int(value)
    except ValueError:
        raise BenchmarkError(f"{combo.key}: invalid {TIME_MARKER} line: {lines[-1]!r}") from None
    if time_ns <= 0:
        raise BenchmarkError(f"{combo.key}: {TIME_MARKER} must be > 0, got {time_ns}")
    return time_ns


# Readable log of a command: what ran, how it ended, and what it printed.
def _log(result: ProcessResult) -> str:
    """Command, exit status and output of a process, for build/test logs and errors."""
    status = "timed out" if result.timed_out else f"exit code {result.returncode}"
    parts = [f"$ {' '.join(result.cmd)}  ({status})"]
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(result.stderr.rstrip())
    return "\n".join(parts)
