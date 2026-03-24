# Durable Gemini Loan Origination

[![CI](https://github.com/samingbar/durable-gemini-loan-origination/actions/workflows/ci.yml/badge.svg)](https://github.com/samingbar/durable-gemini-loan-origination/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/samingbar/durable-gemini-loan-origination)](LICENSE)

Temporal-based mortgage underwriting and insurance claims demos that combine Gemini OCR, policy-grounded analysis, and human review UIs.

> This repository is a demo system built around synthetic data and simplified policy logic. Do not use it for real lending or claims decisions.

## Overview

This repo contains three runnable workflow packages:

| Workflow | Pattern | Best For |
| --- | --- | --- |
| `mortgage_fixed_flow` | Deterministic, fixed-order pipeline | Baseline underwriting behavior and repeatable demos |
| `mortgage_embedded_agent` | Supervisor-driven routing | Comparing fixed orchestration against adaptive sequencing |
| `insurance_claims_fixed_flow` | Deterministic, fixed-order pipeline | A claims-focused variant with a hardened review UI |

If you are new to Temporal, start with `docs/temporal-primitives.md`.

## Repository Layout

- `src/workflows/` contains the workflow packages, their activities, review UIs, and tests.
- `resources/` contains policy PDFs and synthetic test cases.
- `datasets/` contains sample images, generated PDFs, and uploaded review UI cases.
- `docs/` contains Temporal patterns, testing guidance, and workflow authoring notes.

## Quickstart

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

Workers load `.env` from the repository root if it exists. Review UIs inherit the current shell environment, so export variables before starting `uvicorn`.

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

### Start a Worker

Mortgage fixed flow:

```bash
uv run -m src.workflows.mortgage_fixed_flow.worker
```

Mortgage embedded agent:

```bash
uv run -m src.workflows.mortgage_embedded_agent.worker
```

Insurance claims fixed flow:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.worker
```

### Start a Review UI

Mortgage fixed flow:

```bash
uv run uvicorn src.workflows.mortgage_fixed_flow.review_app:app --reload
```

Mortgage embedded agent:

```bash
uv run uvicorn src.workflows.mortgage_embedded_agent.review_app:app --reload
```

Insurance claims fixed flow:

```bash
uv run uvicorn src.workflows.insurance_claims_fixed_flow.review_app:app --reload
```

Open `http://localhost:8000` and upload page images named like `CASEID_p1.png`, `CASEID_p2.png`, and so on.

## Sample Inputs

Run the embedded-agent mortgage demo against the checked-in sample images:

```bash
uv run -m src.workflows.mortgage_embedded_agent.demo --image-dir datasets/images
```

Generate insurance claim PDFs and OCR page images from the checked-in fixtures:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.sample_case_generator
```

The generator writes output under `datasets/insurance_claims/pdfs` and `datasets/insurance_claims/images`. The insurance package does not currently include a separate CLI demo runner; use the review UI or start workflows programmatically.

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

## Common Utilities

Reset the insurance review UI cache manifest:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.reset_cache
```

Reset the manifest and purge uploaded insurance cases:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.reset_cache --purge-uploads
```

## Troubleshooting

- If a workflow will not start, confirm the matching worker is running and `TEMPORAL_ADDRESS` points to the same Temporal server.
- If OCR returns malformed JSON, check worker logs. The workflows use deterministic fallbacks, but the raw model response is still the fastest debugging signal.
- If a review UI shows no cases, verify that its upload root exists and is writable.
- If the insurance review UI reports degraded readiness, check `GET /readyz` and confirm Temporal and the upload root are both available.

## License

MIT. See `LICENSE`.
