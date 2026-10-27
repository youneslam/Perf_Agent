"""Shared helper for the stub scripts: read the markers of the dummy Java component."""

import re
from pathlib import Path

COMPONENT = Path("src/main/java/com/example/graph/Component.java")


def source() -> str:
    return COMPONENT.read_text(encoding="utf-8")


def marker(name: str, default: str) -> str:
    match = re.search(rf"//\s*{name}=(\S+)", source())
    return match.group(1) if match else default
