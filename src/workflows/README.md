# Workflows

Each subfolder in this directory is a self-contained workflow package with its own activities, models, utilities, worker, and tests.

**Available workflows**
- `mortgage_fixed_flow` is a deterministic, fixed-order underwriting pipeline.
- `mortgage_embedded_agent` adds a supervisor activity that selects the next specialist to run.

**Where to start**
- For a clean baseline, read `src/workflows/mortgage_fixed_flow/README.md`.
- For the more agentic version, read `src/workflows/mortgage_embedded_agent/README.md`.
