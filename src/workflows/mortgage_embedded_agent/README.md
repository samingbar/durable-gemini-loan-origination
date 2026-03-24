# Mortgage Embedded Agent

This package demonstrates a more agentic mortgage-underwriting workflow. A supervisor activity decides which specialist should run next based on the current state of the case and the risk signals gathered so far.

## Use This Package When

- You want to compare adaptive routing against the fixed baseline.
- You want a workflow that makes sequencing decisions with model output.
- You want a CLI demo runner for the mortgage flows.

## Package Layout

- `mortgage_workflow.py` orchestrates the supervisor loop and human review gate.
- `mortgage_activities.py` handles OCR, policy retrieval, specialist prompts, and the supervisor call.
- `mortgage_models.py` defines Pydantic inputs and outputs.
- `mortgage_utils.py` contains deterministic metrics, policy checks, and sanitization helpers.
- `review_app.py` is the FastAPI review console.
- `demo.py` runs sample cases from an image directory.
- `worker.py` runs the Temporal worker.
- `tests/` contains activity, utility, and workflow coverage.

## Run Locally

1. Start a Temporal dev server.

```bash
temporal server start-dev
```

2. Start the worker.

```bash
uv run -m src.workflows.mortgage_embedded_agent.worker
```

3. Start the review UI if you want to upload and review cases manually.

```bash
uv run uvicorn src.workflows.mortgage_embedded_agent.review_app:app --reload
```

4. Run the sample demo against the checked-in mortgage images.

```bash
uv run -m src.workflows.mortgage_embedded_agent.demo --image-dir datasets/images
```

The worker and demo use `MORTGAGE_TASK_QUEUE` and `TEMPORAL_ADDRESS` if they are set. The review UI writes uploads to `UPLOAD_ROOT`, which defaults to `datasets/uploads`.

## Review UI Notes

- Upload page images named like `CASEID_p1.png`, `CASEID_p2.png`, and so on.
- The UI starts a workflow when the upload completes successfully.
- Human review is enabled only when the workflow is still running and the current recommendation is `CONDITIONAL`.

## Sample Data

- `datasets/images` contains checked-in sample OCR page images.
- `resources/mortgage_test_cases.json` contains the structured synthetic mortgage fixtures used by tests.

## Run Tests

```bash
uv run poe test -- src/workflows/mortgage_embedded_agent/tests
```
