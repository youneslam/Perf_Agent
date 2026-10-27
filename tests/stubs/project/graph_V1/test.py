"""Stub test suite: fails when the component contains `// BREAK_TEST`."""

import sys

from stub_component import source

if "// BREAK_TEST" in source():
    print("FAILED ComponentTest.shortestPath: expected 3 but was 4", file=sys.stderr)
    sys.exit(1)
print("Tests run: 3, Failures: 0")
