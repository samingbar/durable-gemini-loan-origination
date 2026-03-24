# Workflow Packages

Each subdirectory in `src/workflows/` is a self-contained workflow package with its own workflow definition, activities, models, review UI, worker, and tests.

## Packages

| Package | Pattern | Use It When | Worker Command | Test Command |
| --- | --- | --- | --- | --- |
| `mortgage_fixed_flow` | Fixed-order mortgage underwriting | You want the simplest, most repeatable baseline | `uv run -m src.workflows.mortgage_fixed_flow.worker` | `uv run poe test -- src/workflows/mortgage_fixed_flow/tests` |
| `mortgage_embedded_agent` | Supervisor-routed mortgage underwriting | You want adaptive specialist ordering and a more agentic flow | `uv run -m src.workflows.mortgage_embedded_agent.worker` | `uv run poe test -- src/workflows/mortgage_embedded_agent/tests` |
| `insurance_claims_fixed_flow` | Fixed-order insurance claim adjudication | You want a deterministic claims pipeline with a more resilient review console | `uv run -m src.workflows.insurance_claims_fixed_flow.worker` | `uv run poe test -- src/workflows/insurance_claims_fixed_flow/tests` |

## Review UI Differences

- The two mortgage packages use `UPLOAD_ROOT` and a lightweight review UI.
- The insurance package uses `INSURANCE_UPLOAD_ROOT` and adds explicit case states, retry support for failed cases, and `GET /healthz` plus `GET /readyz`.

## Where To Start

- Read `src/workflows/mortgage_fixed_flow/README.md` first if you want the cleanest baseline.
- Read `src/workflows/mortgage_embedded_agent/README.md` next if you want to compare a more adaptive orchestration style.
- Read `src/workflows/insurance_claims_fixed_flow/README.md` if you want the claims variant or the hardened review UI.
