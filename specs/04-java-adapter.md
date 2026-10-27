# 04 — Process runner + project adapter

Modules: `perf_agent/infra/process.py`, `perf_agent/adapters/java.py` · Milestone M3
Implements `ProjectAdapterPort`.

The adapter is **command-based**: it never knows how the Java project builds or tests itself.
It only runs the `build_cmd`, `test_cmd` and `run_cmd` given in the scenario, inside a version
folder. The real commands (Maven/Gradle/scripts, Java test classes, benchmark runner) come from
the separate Java repo and are wired later through the scenario file.

## 4.1 Process runner (`infra/process.py`)

```python
class ProcessResult(BaseModel):
    cmd: list[str]; returncode: int; stdout: str; stderr: str
    duration_ns: int; timed_out: bool

def run(cmd: list[str], cwd: Path, timeout_s: float, env: dict[str, str] | None = None) -> ProcessResult
```

- `subprocess.run` with `shell=False`, captured output, `timeout`. On timeout, kill the process
  group and return `timed_out=True` (never raise for timeouts).
- Minimal env: `PATH`, `JAVA_HOME`, `HOME`, `LANG` plus explicit extras. No API keys leak to child processes.
- Truncate stored stdout/stderr to 200 KB each.

## 4.2 Project adapter (`adapters/java.py`)

```python
class BuildResult(BaseModel): ok: bool; log: str; duration_ns: int
class TestResult(BaseModel):  ok: bool; log: str

class JavaAdapter:
    def __init__(self, scenario: Scenario): ...
    def read_component(self, version_dir) -> str
    def write_component(self, version_dir, source: str) -> None   # overwrites the REAL file
    def build(self, version_dir) -> BuildResult                     # build_cmd, ok = exit 0
    def test(self, version_dir) -> TestResult                       # test_cmd,  ok = exit 0
    def run_once(self, version_dir, combo) -> ProcessResult         # run_cmd with {graph},{query}
    def benchmark(self, version_dir, combos) -> BenchmarkResult
```

- `write_component` refuses to write into a folder not ending in `_trial`.
- `write_component` checks the new source declares the same `package` and the same public class
  name (regex on the source, no Java parser needed); otherwise raises `ComponentContractError`
  (the graph treats it like a build failure).
- `benchmark`: for each combo, `warmup_runs` then `measured_runs` executions; wall-clock time per
  run via `time.perf_counter_ns` around the process. Each run must exit 0.
  `output_hash` = sha256 of stdout normalized (strip trailing whitespace per line, drop lines
  starting with `#TIME`). All runs of a combo must have the same hash, else `BenchmarkError`.
- **Contract with the Java repo's runner:** if the program prints a line `#TIME_NS <int>`, use that
  value instead of the process wall time (lets the Java side exclude JVM startup). Everything else
  printed on stdout is the result used for the correctness check.

## 4.3 Stub project for this repo's tests (`tests/stubs/project/graph_V1/`)

No Java here. A fake project that respects the same contract:
- `src/main/java/com/example/graph/Component.java` — a dummy Java file (only read/written as text,
  never compiled) containing `package com.example.graph;` and `public class Component`, plus a
  marker comment `// SPEED=1` and `// RESULT=ok`.
- `build.sh` — exits 1 if the Java file contains `// BREAK_BUILD`, else 0.
- `test.sh` — exits 1 if it contains `// BREAK_TEST`, else 0.
- `run.py {graph} {query}` — prints `result <graph> <query> <RESULT>` and `#TIME_NS <base / SPEED>`
  read from the markers, so a "faster" version is just a file with `// SPEED=2`.

This lets every test (adapter and full graph) run in milliseconds, with no JDK.

## Acceptance criteria

- `run(["sleep","5"], timeout_s=0.2)` returns `timed_out=True` in < 1 s.
- Stub project: build and test OK; `benchmark` returns one `Measurement` per combo with `measured_runs` times.
- `// SPEED=2` halves the measured times; `// RESULT=other` changes the `output_hash`.
- `// BREAK_BUILD` → `BuildResult.ok == False` with the script output in `log`.
- Writing a source with a different class name raises `ComponentContractError`.
- Writing into `graph_V1` raises.
