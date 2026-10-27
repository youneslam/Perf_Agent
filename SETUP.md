# Setup — what you need before starting

## 1. Tools to install on your machine

| Tool | Version | Why | Check |
|---|---|---|---|
| Python | 3.11 or 3.12 | the project language | `python --version` |
| uv | latest | creates the venv, installs and locks dependencies | `uv --version` |
| JDK | 17 or 21 (Temurin) | only for real runs against the Java repo (not for this repo's tests) | `javac -version` |
| Git + a GitHub account | any | the repo for the code (the *tool* does not use git; versions are folders) | `git --version` |
| Docker | any | runs Neo4j locally (milestone M9 only) | `docker --version` |
| Claude Code | latest | writes the code from the specs | `claude --version` |
| Ollama *(optional)* | latest | local/free model if external LLMs are not allowed | `ollama --version` |

Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` (macOS/Linux) or
`powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows).
On Windows, use WSL2 to keep the `build.sh` / `test.sh` scripts working.

## 2. Accounts and keys

| Key | Where | Needed for |
|---|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com | Claude models (Planner/Coder/Critic) |
| `OPENAI_API_KEY` *(optional)* | platform.openai.com | OpenAI models |
| `NEO4J_PASSWORD` | you choose it when starting the container | Neo4j storage (M9) |
| `LANGSMITH_API_KEY` *(optional)* | smith.langchain.com | traces of every LLM call and graph step (great for debugging and for the report) |

Nothing is needed for milestones M1–M8 tests: they use the fake model. Use `--dry-run` for demos without keys.

## 3. Create the repo

```bash
mkdir perf-agent && cd perf-agent
git init
# copy CLAUDE.md, SETUP.md and specs/ from this folder into the repo
uv init --package --name perf-agent --python 3.11
```

## 4. Dependencies

```bash
uv add "langgraph>=1.0,<2" langgraph-checkpoint-sqlite \
       "langchain>=1.0,<2" "langchain-core>=1.0,<2" \
       langchain-anthropic langchain-openai langchain-ollama \
       typer rich "pydantic>=2" pyyaml python-dotenv neo4j
uv add --dev pytest pytest-cov ruff mypy types-PyYAML
```

`pyproject.toml` must also declare:

```toml
[project.scripts]
perf-agent = "perf_agent.cli:app"

[tool.pytest.ini_options]
markers = [
  "neo4j: needs a running Neo4j",
  "live: calls a real LLM API (costs money)",
]
addopts = "-m 'not live and not neo4j'"

[tool.mypy]
strict = true
plugins = ["pydantic.mypy"]

[tool.ruff]
line-length = 100
```

## 5. Environment file

Create `.env` (and add it to `.gitignore`), commit a `.env.example` without values:

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
OLLAMA_HOST=http://localhost:11434
NEO4J_PASSWORD=
# optional tracing
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=perf-agent
```

`.gitignore`: `.env`, `.venv/`, `__pycache__/`, `workspace/`, `*.class`, `out/`, `.perf_agent/`.

## 6. Neo4j (only for milestone M9)

```bash
docker run -d --name perf-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/$NEO4J_PASSWORD neo4j:5
```
Browser UI: http://localhost:7474

## 7. Ollama (optional)

```bash
ollama pull qwen2.5-coder
```
Then set `llm.provider: ollama` in the scenario.

## 8. Wiring the Java repo later (not needed to build the agent)

This repo contains only the agent. The Java project, its tests/benchmarks and the front end
stay in your other repo. When you're ready to connect them, write a scenario YAML whose:
- `workspace` points to a folder containing `graph_V1/` (a copy of the Java project, so the
  agent never touches your working tree);
- `build_cmd` / `test_cmd` / `run_cmd` call the commands from that repo (Maven/Gradle/scripts);
- the runner prints a deterministic result for `{graph}` `{query}`, ideally with `#TIME_NS <n>`
  measured inside Java (spec 04).

The front end reads `workspace/reports/<run_id>/run.json` (spec 10) and/or Neo4j (spec 08).

## 9. Start with Claude Code

```bash
cd perf-agent
claude
```
First prompt:
> Read CLAUDE.md and SETUP.md, then implement milestone M1 from specs/00-overview.md.
> Run ruff, mypy and pytest at the end and fix everything before stopping.

Then one milestone per session: "Implement M2", "Implement M3"… Commit after each green milestone.

## Checklist

- [ ] `python --version` ≥ 3.11, `uv --version`
- [ ] repo created with `CLAUDE.md`, `SETUP.md`, `specs/`
- [ ] dependencies installed with `uv add`
- [ ] `.env` filled, `.env` in `.gitignore`
- [ ] (M9) Neo4j container running
- [ ] (later, wiring) JDK ≥ 17 and the Java repo's commands work by hand from `workspace/graph_V1/`
