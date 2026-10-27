"""Stub build: fails when the component contains `// BREAK_BUILD`."""

import sys

from stub_component import COMPONENT, source

if "// BREAK_BUILD" in source():
    print(f"{COMPONENT}:5: error: ';' expected", file=sys.stderr)
    sys.exit(1)
print("BUILD SUCCESSFUL")
