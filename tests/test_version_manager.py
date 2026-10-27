"""Tests for perf_agent/versions/manager.py (spec 03)."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from perf_agent.versions.manager import VERSION_INFO_FILE, VersionError, VersionManager


def _tree_hash(root: Path) -> str:
    """Hash of every file's relative path and content under `root`."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace with `graph_V1/` holding sources, build output and a `build` Java package."""
    v1 = tmp_path / "graph_V1"
    (v1 / "src/com/example/build").mkdir(parents=True)
    (v1 / "src/com/example/Component.java").write_text("class Component {}", encoding="utf-8")
    (v1 / "src/com/example/Component.class").write_bytes(b"\xca\xfe")
    (v1 / "src/com/example/build/Helper.java").write_text("class Helper {}", encoding="utf-8")
    for build_dir in ("out", "target", "build", ".gradle"):
        (v1 / build_dir).mkdir()
        (v1 / build_dir / "artifact.bin").write_bytes(b"x")
    (v1 / "build.sh").write_text("exit 0\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()
    return tmp_path


@pytest.fixture
def manager(workspace: Path) -> VersionManager:
    return VersionManager(workspace, "graph")


def test_only_v1(manager: VersionManager) -> None:
    assert manager.list_versions() == ["graph_V1"]
    assert manager.latest() == "graph_V1"
    assert manager.next_name() == "graph_V2"


def test_create_trial_copies_sources_without_build_output(
    manager: VersionManager, workspace: Path
) -> None:
    trial = manager.create_trial("graph_V1")

    assert trial == workspace / "graph_V2_trial"
    assert _files(trial) == {
        "src/com/example/Component.java",
        "src/com/example/build/Helper.java",
        "build.sh",
    }


def test_create_trial_keeps_file_permissions(manager: VersionManager, workspace: Path) -> None:
    script = workspace / "graph_V1" / "build.sh"
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    trial = manager.create_trial("graph_V1")

    assert (trial / "build.sh").stat().st_mode == script.stat().st_mode


def test_create_trial_replaces_a_leftover_trial(
    manager: VersionManager, workspace: Path
) -> None:
    leftover = workspace / "graph_V2_trial"
    leftover.mkdir()
    (leftover / "stale.txt").write_text("old", encoding="utf-8")

    trial = manager.create_trial("graph_V1")

    assert not (trial / "stale.txt").exists()


def test_promote_renames_trial_and_writes_version_info(
    manager: VersionManager, workspace: Path
) -> None:
    trial = manager.create_trial("graph_V1")

    name = manager.promote(trial, parent="graph_V1", gain_pct=18.4, idea_title="Use ArrayDeque")

    assert name == "graph_V2"
    assert manager.list_versions() == ["graph_V1", "graph_V2"]
    assert not trial.exists()
    assert (workspace / "graph_V2" / VERSION_INFO_FILE).is_file()
    info = manager.read_info("graph_V2")
    assert info is not None
    assert (info.name, info.parent, info.gain_pct, info.idea_title) == (
        "graph_V2",
        "graph_V1",
        18.4,
        "Use ArrayDeque",
    )
    assert manager.read_info("graph_V1") is None


def test_promote_never_overwrites_an_existing_version(
    manager: VersionManager, workspace: Path
) -> None:
    trial = manager.create_trial("graph_V1")
    (workspace / "graph_V2").mkdir()

    with pytest.raises(VersionError, match="already exists"):
        manager.promote(trial, parent="graph_V1", gain_pct=5, idea_title="x")


def test_promote_missing_trial_raises(manager: VersionManager, workspace: Path) -> None:
    with pytest.raises(VersionError, match="trial folder not found"):
        manager.promote(workspace / "graph_V2_trial", parent="graph_V1", gain_pct=5, idea_title="x")


def test_discard_deletes_the_trial(manager: VersionManager) -> None:
    trial = manager.create_trial("graph_V1")

    manager.discard(trial)

    assert not trial.exists()
    assert manager.list_trials() == []


def test_discard_a_missing_trial_does_nothing(manager: VersionManager, workspace: Path) -> None:
    manager.discard(workspace / "graph_V9_trial")


def test_discard_deletes_read_only_files(manager: VersionManager) -> None:
    trial = manager.create_trial("graph_V1")
    locked = trial / "src/com/example/Component.java"
    os.chmod(locked, stat.S_IREAD)

    manager.discard(trial)

    assert not trial.exists()


@pytest.mark.parametrize(
    "target",
    [
        "graph_V1",
        "reports",
        "graph_V1/../graph_V1",
        "../outside_V2_trial",
        "graph_V1/graph_V2_trial",
        "other_V2_trial",
    ],
)
def test_discard_refuses_anything_but_a_trial_of_the_workspace(
    manager: VersionManager, workspace: Path, target: str
) -> None:
    with pytest.raises(VersionError, match="refusing to touch"):
        manager.discard(workspace / target)


def test_original_is_unchanged_after_a_full_cycle(
    manager: VersionManager, workspace: Path
) -> None:
    before = _tree_hash(workspace / "graph_V1")

    accepted = manager.create_trial("graph_V1")
    (accepted / "src/com/example/Component.java").write_text("faster", encoding="utf-8")
    manager.promote(accepted, parent="graph_V1", gain_pct=10, idea_title="faster")
    rejected = manager.create_trial("graph_V2")
    (rejected / "src/com/example/Component.java").write_text("slower", encoding="utf-8")
    manager.discard(rejected)

    assert _tree_hash(workspace / "graph_V1") == before


def test_numeric_ordering(tmp_path: Path) -> None:
    for n in (10, 2, 1, 9):
        (tmp_path / f"graph_V{n}").mkdir()
    (tmp_path / "graph_V3_trial").mkdir()
    (tmp_path / "graph_v4").mkdir()
    (tmp_path / "graph_V5.txt").write_text("not a folder", encoding="utf-8")
    manager = VersionManager(tmp_path, "graph")

    assert manager.list_versions() == ["graph_V1", "graph_V2", "graph_V9", "graph_V10"]
    assert manager.latest() == "graph_V10"
    assert manager.next_name() == "graph_V11"


def test_cleanup_stale_trials(manager: VersionManager, workspace: Path) -> None:
    for name in ("graph_V3_trial", "graph_V2_trial"):
        (workspace / name).mkdir()

    removed = manager.cleanup_stale_trials()

    assert removed == [workspace / "graph_V2_trial", workspace / "graph_V3_trial"]
    assert manager.list_trials() == []
    assert manager.list_versions() == ["graph_V1"]


@pytest.mark.parametrize(
    ("name", "message"),
    [("graph_V7", "version folder not found"), ("graphV1", "is not a version name")],
)
def test_check_base_errors(manager: VersionManager, name: str, message: str) -> None:
    with pytest.raises(VersionError, match=message):
        manager.check_base(name)


def test_check_base_returns_the_folder(manager: VersionManager, workspace: Path) -> None:
    assert manager.check_base("graph_V1") == workspace / "graph_V1"


def test_missing_workspace(tmp_path: Path) -> None:
    manager = VersionManager(tmp_path / "missing", "graph")

    with pytest.raises(VersionError, match="workspace folder not found"):
        manager.list_versions()


def test_latest_without_versions(tmp_path: Path) -> None:
    with pytest.raises(VersionError, match="no graph_V<n> folder"):
        VersionManager(tmp_path, "graph").latest()


def test_custom_prefix_and_ignore(tmp_path: Path) -> None:
    (tmp_path / "bfs_V1" / "dist").mkdir(parents=True)
    (tmp_path / "bfs_V1" / "dist" / "a.jar").write_bytes(b"x")
    (tmp_path / "bfs_V1" / "notes.tmp").write_text("x", encoding="utf-8")
    (tmp_path / "bfs_V1" / "A.class").write_bytes(b"x")
    manager = VersionManager(tmp_path, "bfs", ignore=["dist/", "*.tmp"])

    trial = manager.create_trial("bfs_V1")

    assert trial.name == "bfs_V2_trial"
    assert _files(trial) == {"A.class"}
