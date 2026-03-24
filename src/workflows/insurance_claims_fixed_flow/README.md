# Insurance Claims Fixed Flow

This package is the deterministic insurance-claims baseline. It follows a fixed pipeline: OCR intake, policy retrieval, specialist analyses, critic review, structured decision memo generation, and a human review gate.

## Use This Package When

- You want stable, repeatable claim adjudication behavior.
- You want a clean OCR-plus-LLM baseline before experimenting with agentic routing.
- You want the most capable review UI in this repository.

## Package Layout

- `insurance_workflow.py` orchestrates the fixed-order workflow and human review gate.
- `insurance_activities.py` handles OCR, policy retrieval, and model calls.
- `insurance_models.py` defines Pydantic inputs and outputs.
- `insurance_utils.py` contains deterministic metrics, risk flags, and policy checks.
- `review_app.py` is the FastAPI review console.
- `worker.py` runs the Temporal worker.
- `utils/sample_case_generator.py` generates sample claim PDFs and OCR page images.
- `utils/reset_cache.py` clears the review UI cache manifest and, optionally, uploaded cases.
- `tests/` contains activity, utility, workflow, reset-cache, and review-app coverage.

## Run Locally

1. Start a Temporal dev server.

```bash
temporal server start-dev
```

2. Start the worker.

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.worker
```

3. Start the review UI.

```bash
uv run uvicorn src.workflows.insurance_claims_fixed_flow.review_app:app --reload
```

4. Upload claim page images named like `CASEID_p1.png`, `CASEID_p2.png`, and so on.

The worker uses `INSURANCE_TASK_QUEUE` and `TEMPORAL_ADDRESS` if they are set. The review UI writes uploads to `INSURANCE_UPLOAD_ROOT`, which defaults to `datasets/uploads/insurance_claims`.

## Review UI Behavior

- Uploads are validated before workflow start.
- The UI persists explicit case states: `QUEUED`, `RUNNING`, `AWAITING_REVIEW`, `COMPLETED`, `FAILED`, and `START_FAILED`.
- Failed and start-failed cases can be retried from the case page.
- `GET /healthz` reports process health.
- `GET /readyz` reports Temporal connectivity and upload-root readiness.
- When Temporal is unavailable, the UI stays online in degraded mode and disables actions that depend on Temporal.

Default insurance upload limits:

- `INSURANCE_MAX_FILES=10`
- `INSURANCE_MAX_FILE_BYTES=10485760`
- `INSURANCE_MAX_TOTAL_BYTES=26214400`

## Generate Sample Cases

Generate the checked-in insurance fixtures as PDFs and page images:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.sample_case_generator
```

Output is written to:

- `datasets/insurance_claims/pdfs`
- `datasets/insurance_claims/images`

## Reset The Review Cache

Clear only the insurance case manifest:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.reset_cache
```

Clear the manifest and uploaded insurance cases:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.utils.reset_cache --purge-uploads
```

## Run Tests

```bash
uv run poe test -- src/workflows/insurance_claims_fixed_flow/tests
```

## Data And Policy Inputs

- `resources/insurance_claim_test_cases.json` contains the synthetic claim fixtures.
- `resources/insurance_claim_policies.pdf` is the default policy corpus used for grounding.
- `datasets/insurance_claims/` contains generated sample PDFs and OCR images.
