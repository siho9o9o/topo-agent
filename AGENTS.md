# AGENTS.md

Instructions for coding agents working in this repository. Follow this file on every task unless the user overrides it.

## Project

`topo-agent`: tap one hold in a bouldering-gym photo, get every hold of the same boulder problem.
SAM 3.1 segments hold candidates, then a Muse Spark agent (`muse-spark-1.3`) uses tools
(`inspect_candidates`, `find_similar`, `look_closer`, `submit_holds`) to pick the ones in the tapped problem.
Results are shared as problems in the Mirai Climbing app. See [README.md](README.md) for the full picture.

## Language

All code, comments, docstrings, docs, and commit messages must be in English. Never add non-English text to the repo.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e .          # add [gemini] for the Gemini baseline
```

Python >= 3.10. Runtime dependencies live in `pyproject.toml` (`pillow`, `httpx`, `meta-sam-parser`).

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

- Tests use a fake SAM and a fake Muse Spark: no network and no API keys needed. Keep it that way.
- Every behavior change or bug fix needs a matching test in `tests/` (same `test_*.py` naming).
- Run the full suite before finishing; all tests must pass. Never narrow, skip, or weaken a test to make it pass.

## Running with real APIs

API keys come from the environment (a git-ignored `.env` file with `KEY=VALUE` lines also works for `web.py`).
Never print keys or commit them.

```bash
MODEL_API_KEY=... .venv/bin/python -m topo_agent.sam photo.jpg "climbing hold"
MODEL_API_KEY=... .venv/bin/python -m topo_agent.eval --bench --methods color,sam_color,meta --seeds 11
.venv/bin/python -m topo_agent.web   # local GUI on http://127.0.0.1:8000 (requests logged to out/web.log)
```

- `eval.py --bench` uses code-drawn sample walls, which may be sent to external AI APIs.
- Tuning history lives in [docs/experiments.md](docs/experiments.md); check it before re-trying an optimization.
- Real gym photos may only be sent to external APIs with the photographer's consent.
- API requests must use `store: false` and must not include names or user data.

## Conventions

- New Python files start with `# SPDX-License-Identifier: AGPL-3.0-or-later` (see [COPYING](COPYING)).
- Prefer the standard library; do not add dependencies without asking.
- Keep helpers deterministic and testable without network access (inject `post`/`model_call` fakes, as in `sam.py` and `agent.py`).
- There is no linter, formatter, or CI config; match the surrounding style (short functions, plain docstrings).
- Write run artifacts (reports, annotated images, traces) under `out/` (git-ignored), never into the source tree.

## Do not touch

- `topo_agent/baseline/` is pre-hackathon code (tagged `pre-hackathon`), kept only as a comparison baseline.
  Do not modify it; do not move its logic elsewhere.
- `topo_agent/eval.py` scoring semantics (recall/precision/match rules) are the shared contract with the app repo;
  changing them needs explicit user approval.

## Upstream tracking

The original project is https://github.com/Mirai-Climbing/topo-agent. This clone's `origin` is a personal
fork, so from time to time (at the start of a non-trivial task, or before touching files shared with upstream):

- Make sure a remote points at the original (`git remote -v`; add it once with
  `git remote add upstream https://github.com/Mirai-Climbing/topo-agent.git` if missing), then
  `git fetch upstream` and compare (`git log HEAD..upstream/main --oneline`, `git diff HEAD upstream/main --stat`).
- If upstream moved, tell the user what changed and ask before merging or rebasing. Never push to upstream.
- Keep the `pre-hackathon` tag and `topo_agent/baseline/` semantics aligned with upstream; do not invent divergences.

## File map

- `topo_agent/sam.py` — SAM 3.1 segmentation (Responses API, mask parsing, candidates)
- `topo_agent/agent.py` — Muse Spark agent loop, tools, `sam_color`/`meta` finders, fallbacks
- `topo_agent/eval.py` — scoring (`evaluate`/`summary`) and `--bench` CLI
- `topo_agent/web.py` + `topo_agent/static/index.html` — local demo GUI (stdlib only)
- `topo_agent/baseline/` — pre-hackathon: color-only finder, sample walls (`bench.py`), Gemini agent
- `tests/` — unit tests, all offline with fakes
- `docs/` — experiment log and other docs
