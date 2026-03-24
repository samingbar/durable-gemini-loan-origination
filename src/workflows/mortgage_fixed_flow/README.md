# Mortgage Fixed Flow

This package is the deterministic mortgage-underwriting baseline. It always runs the same stages in the same order, which makes it the easiest flow to reason about, test, and compare against more adaptive variants.

## Use This Package When

- You want the simplest underwriting pipeline in the repository.
- You want repeatable, fixed-order behavior for demos or regression testing.
- You want a clear reference implementation before reading the embedded-agent version.

## Key Files

- `mortgage_workflow.py` orchestrates OCR intake, specialist analysis, memo generation, and the human review gate.
- `mortgage_activities.py` handles OCR, policy retrieval, and specialist prompts.
- `mortgage_models.py` defines Pydantic inputs and outputs.
- `mortgage_utils.py` contains deterministic metrics, policy checks, and sanitization helpers.
- `review_app.py` is the FastAPI review console.
- `worker.py` runs the Temporal worker.
- `tests/` contains activity, utility, workflow, and review-app coverage.

## Run Locally

Set `GEMINI_API_KEY` before starting the flow. You can also set `TEMPORAL_ADDRESS`, `MORTGAGE_TASK_QUEUE`, and `UPLOAD_ROOT` if you need overrides.

1. Start a Temporal dev server.

```bash
temporal server start-dev
```

2. Start the worker.

```bash
uv run -m src.workflows.mortgage_fixed_flow.worker
```

3. Start the review UI.

```bash
uv run uvicorn src.workflows.mortgage_fixed_flow.review_app:app --reload
```

4. Open `http://localhost:8000` and upload mortgage page images named like `CASEID_p1.png`, `CASEID_p2.png`, and so on.

Uploads are written under `UPLOAD_ROOT`, which defaults to `datasets/uploads`.

## Review UI Behavior

- The UI starts a workflow immediately after upload.
- Review submission is allowed only while the workflow is running and the current recommendation is `CONDITIONAL`.
- This UI is intentionally simpler than the insurance review console. It does not include the insurance flow's degraded-mode, retry, or health-check behavior.

## Sample Inputs

There is no dedicated CLI demo runner for this package. Use the review UI with your own page images or with the checked-in mortgage sample images in `datasets/images`.

## Run Tests

```bash
uv run poe test -- src/workflows/mortgage_fixed_flow/tests
```

## Data And Policy Inputs

- `resources/mortgage_test_cases.json` contains the structured synthetic mortgage fixtures.
- `resources/underwriting_policies.pdf` is the default policy corpus used for grounding.

## Compare With

If you want to see the adaptive variant next, read [`../mortgage_embedded_agent/README.md`](../mortgage_embedded_agent/README.md).
