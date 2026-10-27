# 03 — Version manager (replaces git worktrees)

Module: `perf_agent/versions/manager.py` · Milestone M2 · Implements `VersionManagerPort`

## Concept

Every version is a **complete copy** of the project on disk:

```
workspace/
  graph_V1/          # original, never modified
  graph_V2/          # promoted after trial accepted
  graph_V3_trial/    # current trial (temporary)
```

The Java class keeps its real name in every folder; only the folder name changes.

## API

```python
class VersionManager:
    def __init__(self, workspace: Path, prefix: str, ignore: list[str] | None = None): ...
    def list_versions(self) -> list[str]            # promoted only, numeric order
    def latest(self) -> str
    def next_name(self) -> str                       # graph_V<max+1>
    def check_base(self, name: str) -> Path          # raises VersionError if missing
    def create_trial(self, base: str) -> Path        # copies base → <next_name>_trial
    def promote(self, trial_dir: Path) -> str        # renames <x>_trial → <x>, returns name
    def discard(self, trial_dir: Path) -> None       # deletes a *_trial folder
    def cleanup_stale_trials(self) -> list[Path]     # deletes leftover *_trial at startup
```

## Requirements

- `create_trial` uses `shutil.copytree` excluding `ignore` patterns (default: `out/`, `target/`,
  `build/`, `*.class`, `.gradle/`) so the build starts clean. Preserve file permissions (scripts stay executable).
- If a trial folder already exists for that name, delete it first and log a warning.
- `promote`: atomic `Path.rename` (same filesystem). Fails if the target name already exists.
  After promotion, mark the folder read-only is NOT required, but nothing in the code may write to it.
- `discard` refuses (raises `VersionError`) any path that does not end in `_trial` or is outside the workspace.
- Version name parsing: `^<prefix>_V(\d+)$`; trial: `^<prefix>_V(\d+)_trial$`.
- Write a small `VERSION.json` in each promoted folder: `{name, parent, created_at, gain_pct, idea_title}`
  (written while still a trial, just before rename).

## Acceptance criteria

- With `graph_V1` only: `next_name() == "graph_V2"`; `create_trial("graph_V1")` creates `graph_V2_trial` with identical sources minus ignored files.
- `promote` then `list_versions() == ["graph_V1", "graph_V2"]`.
- `discard` on `graph_V1` raises; on a path outside workspace raises.
- Numeric ordering: V2 < V10.
- `cleanup_stale_trials` removes leftover trial folders and returns them.
- Original `graph_V1` content hash is unchanged after a full create/promote/discard cycle.
