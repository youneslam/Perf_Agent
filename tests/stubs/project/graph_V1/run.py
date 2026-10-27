"""Stub benchmark runner: `run.py <graph> <query>`.

Prints `result <graph> <query> <RESULT>` and `#TIME_NS <1_000_000 / SPEED>`, both read from
the component markers, so a "faster" version is just a file with `// SPEED=2`.
`// CRASH` makes the run exit 1.
"""

import sys

from stub_component import marker, source

BASE_TIME_NS = 1_000_000

graph, query = sys.argv[1], sys.argv[2]
if "// CRASH" in source():
    print("Exception in thread main: java.lang.NullPointerException", file=sys.stderr)
    sys.exit(1)
print(f"result {graph} {query} {marker('RESULT', 'ok')}")
print(f"#TIME_NS {BASE_TIME_NS // int(marker('SPEED', '1'))}")
