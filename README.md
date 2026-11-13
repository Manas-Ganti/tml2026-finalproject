> **Branch `manas/rl`: personal GRPO portfolio work, not part of the course deliverable.** Design: `docs/rl_canary_design.md`. Never merge this branch into `main` or `manas/mvr-pipeline`.

# Data Provenance Canary

Trustworthy ML final project (Fall 2026). How many hidden canaries does a writer need to prove an AI model
was trained on their work? We compare four canary types across copy counts, data-cleaning steps and
model-access levels, using Pythia models we train ourselves.

- **Plan and team rules:** [`CLAUDE.md`](CLAUDE.md). Read this first; every teammate's Claude Code agent reads it too.
- **Abstract (approved):** [`docs/abstract.md`](docs/abstract.md)
- **Phase plan, roles, tasks:** [`docs/PLAN.md`](docs/PLAN.md)
- **Agent message board:** [`docs/BOARD.md`](docs/BOARD.md) (`scripts/board.sh inbox`)
- **Status log:** [`docs/STATUS.md`](docs/STATUS.md). Append an entry after each work session.
- **Visual overview:** open `project-overview.html` in a browser.
- **Original proposal:** `Data Provenance Canary Project Proposal.pdf`

## Working together
Branch from `main` as `<name>/<topic>`, open a PR, and get one teammate's review before merging.
Never commit data, checkpoints or tokens. Full rules are in CLAUDE.md §3b.

## Quick start
```bash
python -m pip install -e ".[dev]"
python -m pytest -q                         # interface + fixture tests
scripts/board.sh init <yourname>-agent      # once, then: scripts/board.sh inbox
```
