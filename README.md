# Durable Gemini Loan Origination

[![CI](https://github.com/samingbar/durable-gemini-loan-origination/actions/workflows/ci.yml/badge.svg)](https://github.com/samingbar/durable-gemini-loan-origination/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/samingbar/durable-gemini-loan-origination)](LICENSE)

Temporal demos for mortgage underwriting and insurance claim adjudication built with Gemini OCR, policy-grounded analysis, and human review UIs.

> This repository is a demo system built around synthetic data and simplified policy logic. Do not use it for real lending or claims decisions.

## Choose A Workflow

This repository currently ships three runnable workflow packages:

| Workflow | Pattern | Best for |
| --- | --- | --- |
| [`mortgage_fixed_flow`](src/workflows/mortgage_fixed_flow/README.md) | Deterministic, fixed-order mortgage pipeline | Baseline underwriting behavior, regression testing, and the simplest review flow |
| [`mortgage_embedded_agent`](src/workflows/mortgage_embedded_agent/README.md) | Supervisor-routed mortgage pipeline | Comparing fixed orchestration against adaptive specialist sequencing |
| [`insurance_claims_fixed_flow`](src/workflows/insurance_claims_fixed_flow/README.md) | Deterministic, fixed-order claims pipeline | Claims adjudication demos and the most capable review UI in the repo |

For a package-level index, read [`src/workflows/README.md`](src/workflows/README.md). If you are new to Temporal, start with [`docs/temporal-primitives.md`](docs/temporal-primitives.md).

## Repository Layout

- `src/workflows/` contains the runnable workflow packages, their review UIs, workers, and tests.
- `resources/` contains policy documents and structured synthetic fixtures.
- `datasets/` contains sample OCR images, generated PDFs, and uploaded review UI cases.
- `docs/` contains Temporal patterns, testing guidance, and workflow authoring notes.

## Quick Start

### Prerequisites

- Python 3.12+
- `uv`
- Temporal CLI
- A Gemini API key

### Install Dependencies

```bash
uv sync --dev
```

### Configure Environment

Export variables before starting workers, demos, or `uvicorn`. Some worker and demo scripts also load a repo-level `.env` for runtime values such as API keys, but the review UIs use whatever is already present in the shell at startup.

```bash
export GEMINI_API_KEY="your_api_key"
export TEMPORAL_ADDRESS="localhost:7233"

# Optional overrides
export GEMINI_MODEL="your-preferred-model"
export MORTGAGE_TASK_QUEUE="mortgage-underwriting"
export INSURANCE_TASK_QUEUE="insurance-claims"
export UPLOAD_ROOT="datasets/uploads"
export INSURANCE_UPLOAD_ROOT="datasets/uploads/insurance_claims"
export INSURANCE_POLICY_PATH="resources/insurance_claim_policies.pdf"

# Insurance review UI upload limits
export INSURANCE_MAX_FILES="10"
export INSURANCE_MAX_FILE_BYTES="10485760"
export INSURANCE_MAX_TOTAL_BYTES="26214400"
```

### Start Temporal

```bash
temporal server start-dev
```

### Start A Worker

Pick the workflow you want to run:

```bash
uv run -m src.workflows.mortgage_fixed_flow.worker
uv run -m src.workflows.mortgage_embedded_agent.worker
uv run -m src.workflows.insurance_claims_fixed_flow.worker
```

### Start A Review UI

Pick the matching review UI:

```bash
uv run uvicorn src.workflows.mortgage_fixed_flow.review_app:app --reload
uv run uvicorn src.workflows.mortgage_embedded_agent.review_app:app --reload
uv run uvicorn src.workflows.insurance_claims_fixed_flow.review_app:app --reload
```

Open `http://localhost:8000` and upload page images named like `CASEID_p1.png`, `CASEID_p2.png`, and so on.

## Sample Runs And Utilities

Run the embedded-agent mortgage demo against the checked-in sample images:

```bash
uv run -m src.workflows.mortgage_embedded_agent.demo --image-dir datasets/images
```

Generate the checked-in insurance fixtures as PDFs and page images:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.sample_case_generator
```

The generator writes output under `datasets/insurance_claims/pdfs` and `datasets/insurance_claims/images`.

Reset the insurance review UI cache manifest:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.reset_cache
```

Reset the manifest and purge uploaded insurance cases:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.reset_cache --purge-uploads
```

The insurance package does not currently include a standalone demo runner; use the review UI or start workflows programmatically.

## Review UI Notes

- Mortgage review UIs use `UPLOAD_ROOT`, defaulting to `datasets/uploads`.
- The insurance review UI uses `INSURANCE_UPLOAD_ROOT`, defaulting to `datasets/uploads/insurance_claims`.
- The insurance review UI persists explicit case states: `QUEUED`, `RUNNING`, `AWAITING_REVIEW`, `COMPLETED`, `FAILED`, and `START_FAILED`.
- Failed insurance cases can be retried from the UI.
- The insurance review UI exposes `GET /healthz` and `GET /readyz` and stays online in degraded mode when Temporal is unavailable.

## Testing And Quality

Run the full test suite:

```bash
uv run poe test
```

Run a workflow-specific suite:

```bash
uv run poe test -- src/workflows/mortgage_fixed_flow/tests
uv run poe test -- src/workflows/mortgage_embedded_agent/tests
uv run poe test -- src/workflows/insurance_claims_fixed_flow/tests
```

Lint and format:

```bash
uv run poe lint
uv run poe format
```

## Troubleshooting

- If a workflow will not start, confirm the matching worker is running and `TEMPORAL_ADDRESS` points to the same Temporal server.
- If a review UI cannot connect, re-export the environment variables before starting `uvicorn`.
- If OCR returns malformed JSON, check worker logs. The workflows include deterministic fallbacks, but the raw model response is still the fastest debugging signal.
- If a review UI shows no cases, verify that its upload root exists and is writable.
- If the insurance review UI reports degraded readiness, check `GET /readyz` and confirm Temporal and the upload root are both available.

## License

MIT. See `LICENSE`.
