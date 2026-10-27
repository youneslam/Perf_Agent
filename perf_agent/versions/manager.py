"""Version folders: every version is a complete copy of the project (spec 03).

```
workspace/
  graph_V1/          # original, never modified
  graph_V2/          # promoted after a trial was accepted
  graph_V3_trial/    # current trial (temporary, the only folder that is written to)
```
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

VERSION_INFO_FILE = "VERSION.json"

# Entries ending with "/" are directory names skipped at the root of the version folder only
# (build output); other entries are file-name patterns skipped at any depth.
DEFAULT_IGNORE = ["out/", "target/", "build/", ".gradle/", "*.class"]


class VersionError(Exception):
    """A version folder is missing, badly named, already exists or may not be touched."""


class VersionInfo(BaseModel):
    """Content of `VERSION.json`, written in every promoted folder."""

    model_config = ConfigDict(extra="forbid")

    name: str
    parent: str
    created_at: datetime
    gain_pct: float
    idea_title: str


class VersionManager:
    """Creates, promotes and discards version folders inside one workspace."""

    def __init__(self, workspace: Path, prefix: str, ignore: list[str] | None = None) -> None:
        self._workspace = workspace.resolve()
        self._prefix = prefix
        patterns = DEFAULT_IGNORE if ignore is None else ignore
        self._ignored_root_dirs = {p.rstrip("/") for p in patterns if p.endswith("/")}
        self._ignored_files = [p for p in patterns if not p.endswith("/")]
        self._version_re = re.compile(rf"^{re.escape(prefix)}_V(\d+)$")
        self._trial_re = re.compile(rf"^{re.escape(prefix)}_V(\d+)_trial$")

    @property
    def workspace(self) -> Path:
        return self._workspace

    def list_versions(self) -> list[str]:
        """Promoted versions only, in numeric order (`graph_V2` before `graph_V10`)."""
        return self._names_matching(self._version_re)

    def list_trials(self) -> list[str]:
        """Leftover or current `*_trial` folders, in numeric order."""
        return self._names_matching(self._trial_re)

    def latest(self) -> str:
        versions = self.list_versions()
        if not versions:
            raise VersionError(f"no {self._prefix}_V<n> folder in {self._workspace}")
        return versions[-1]

    def next_name(self) -> str:
        """`<prefix>_V<max+1>`, where max is the highest promoted version number."""
        numbers = [self._number(name) for name in self.list_versions()]
        return f"{self._prefix}_V{max(numbers, default=0) + 1}"

    def check_base(self, name: str) -> Path:
        """Return the folder of version `name`; raise `VersionError` if it is missing."""
        if not self._version_re.match(name):
            raise VersionError(f"{name!r} is not a version name like {self._prefix}_V1")
        path = self._workspace / name
        if not path.is_dir():
            raise VersionError(f"version folder not found: {path}")
        return path

    def create_trial(self, base: str) -> Path:
        """Copy version `base` to `<next_name>_trial/`, without build output."""
        source = self.check_base(base)
        trial = self._workspace / f"{self.next_name()}_trial"
        if trial.exists():
            logger.warning("deleting leftover trial folder %s", trial)
            _remove_tree(trial)
        try:
            shutil.copytree(source, trial, ignore=self._ignore_for(source))
        except OSError as exc:
            raise VersionError(f"cannot copy {source} to {trial}: {exc}") from exc
        logger.info("created trial %s from %s", trial.name, base)
        return trial

    def promote(self, trial_dir: Path, parent: str, gain_pct: float, idea_title: str) -> str:
        """Write `VERSION.json`, then rename `<x>_trial/` to `<x>/`; return `<x>`."""
        trial = self._own_trial(trial_dir)
        if not trial.is_dir():
            raise VersionError(f"trial folder not found: {trial}")
        name = trial.name.removesuffix("_trial")
        target = self._workspace / name
        if target.exists():
            raise VersionError(f"cannot promote {trial.name}: {target} already exists")
        info = VersionInfo(
            name=name,
            parent=parent,
            created_at=datetime.now(UTC),
            gain_pct=gain_pct,
            idea_title=idea_title,
        )
        try:
            (trial / VERSION_INFO_FILE).write_text(info.model_dump_json(indent=2), encoding="utf-8")
            trial.rename(target)
        except OSError as exc:
            raise VersionError(f"cannot promote {trial} to {target}: {exc}") from exc
        logger.info("promoted %s to %s", trial.name, name)
        return name

    def discard(self, trial_dir: Path) -> None:
        """Delete a `*_trial` folder of this workspace. Anything else raises `VersionError`."""
        trial = self._own_trial(trial_dir)
        if not trial.exists():
            logger.debug("trial folder %s already gone", trial)
            return
        _remove_tree(trial)
        logger.info("discarded %s", trial.name)

    def cleanup_stale_trials(self) -> list[Path]:
        """Delete every leftover `*_trial` folder (e.g. after a crash) and return them."""
        removed = []
        for name in self.list_trials():
            path = self._workspace / name
            logger.warning("removing stale trial folder %s", path)
            _remove_tree(path)
            removed.append(path)
        return removed

    def read_info(self, name: str) -> VersionInfo | None:
        """The `VERSION.json` of a promoted version, or None (e.g. for the original V1)."""
        path = self.check_base(name) / VERSION_INFO_FILE
        if not path.is_file():
            return None
        return VersionInfo.model_validate_json(path.read_text(encoding="utf-8"))

    def _names_matching(self, pattern: re.Pattern[str]) -> list[str]:
        if not self._workspace.is_dir():
            raise VersionError(f"workspace folder not found: {self._workspace}")
        names = [
            entry.name
            for entry in self._workspace.iterdir()
            if entry.is_dir() and pattern.match(entry.name)
        ]
        return sorted(names, key=self._number)

    def _number(self, name: str) -> int:
        match = self._version_re.match(name) or self._trial_re.match(name)
        if match is None:
            raise VersionError(f"{name!r} is not a version or trial name")
        return int(match.group(1))

    def _own_trial(self, trial_dir: Path) -> Path:
        """Resolve `trial_dir` and check it is a `*_trial` folder directly in the workspace."""
        resolved = trial_dir.resolve()
        if resolved.parent != self._workspace or not self._trial_re.match(resolved.name):
            raise VersionError(
                f"refusing to touch {trial_dir}: not a {self._prefix}_V<n>_trial folder "
                f"of {self._workspace}"
            )
        return resolved

    def _ignore_for(self, root: Path) -> Callable[[str, list[str]], set[str]]:
        file_patterns = shutil.ignore_patterns(*self._ignored_files)

        def ignore(directory: str, names: list[str]) -> set[str]:
            skipped = set(file_patterns(directory, names))
            if Path(directory) == root:
                skipped |= {
                    n for n in names if n in self._ignored_root_dirs and (root / n).is_dir()
                }
            return skipped

        return ignore


def _remove_tree(path: Path) -> None:
    """`shutil.rmtree` that also deletes read-only files (common on Windows)."""

    def make_writable_and_retry(func: Callable[..., Any], target: str, _exc: object) -> None:
        os.chmod(target, stat.S_IWRITE)
        func(target)

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=make_writable_and_retry)
        else:
            shutil.rmtree(path, onerror=make_writable_and_retry)
    except OSError as exc:
        raise VersionError(f"cannot delete {path}: {exc}") from exc
