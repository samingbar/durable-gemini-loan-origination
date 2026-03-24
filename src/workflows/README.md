# Workflow Packages

Each subdirectory in `src/workflows/` is a self-contained workflow package with its own workflow definition, activities, models, worker, review UI, tests, and package README.

## Packages

| Package | Pattern | Use it when | Docs | Worker command | Test command |
| --- | --- | --- | --- | --- | --- |
| `mortgage_fixed_flow` | Fixed-order mortgage underwriting | You want the simplest, most repeatable baseline | [README](mortgage_fixed_flow/README.md) | `uv run -m src.workflows.mortgage_fixed_flow.worker` | `uv run poe test -- src/workflows/mortgage_fixed_flow/tests` |
| `mortgage_embedded_agent` | Supervisor-routed mortgage underwriting | You want adaptive specialist ordering and a more agentic flow | [README](mortgage_embedded_agent/README.md) | `uv run -m src.workflows.mortgage_embedded_agent.worker` | `uv run poe test -- src/workflows/mortgage_embedded_agent/tests` |
| `insurance_claims_fixed_flow` | Fixed-order insurance claim adjudication | You want a deterministic claims pipeline with a more resilient review console | [README](insurance_claims_fixed_flow/README.md) | `uv run -m src.workflows.insurance_claims_fixed_flow.worker` | `uv run poe test -- src/workflows/insurance_claims_fixed_flow/tests` |

## Shared Conventions

- All packages connect to Temporal via `TEMPORAL_ADDRESS`.
- The two mortgage packages use `MORTGAGE_TASK_QUEUE` and `UPLOAD_ROOT`.
- The insurance package uses `INSURANCE_TASK_QUEUE` and `INSURANCE_UPLOAD_ROOT`.
- Only `mortgage_embedded_agent` includes a dedicated CLI demo runner.
- Only the insurance package includes cache-reset and sample-case generation utilities.

## Where To Start

- Read [`mortgage_fixed_flow/README.md`](mortgage_fixed_flow/README.md) first if you want the cleanest baseline.
- Read [`mortgage_embedded_agent/README.md`](mortgage_embedded_agent/README.md) next if you want to compare adaptive routing against the fixed baseline.
- Read [`insurance_claims_fixed_flow/README.md`](insurance_claims_fixed_flow/README.md) if you want the claims variant or the hardened review UI.
