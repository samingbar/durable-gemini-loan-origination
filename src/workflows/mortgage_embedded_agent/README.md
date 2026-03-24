# Mortgage Embedded Agent

This package is the more agentic mortgage-underwriting flow in the repository. A supervisor activity decides which specialist should run next based on the current case state and the risk signals gathered so far.

## Use This Package When

- You want to compare adaptive routing against the fixed baseline.
- You want a workflow that makes sequencing decisions with model output.
- You want the mortgage package with a built-in CLI demo runner.

## Key Files

- `mortgage_workflow.py` orchestrates the supervisor loop and human review gate.
- `mortgage_activities.py` handles OCR, policy retrieval, specialist prompts, and the supervisor call.
- `mortgage_models.py` defines Pydantic inputs and outputs.
- `mortgage_utils.py` contains deterministic metrics, policy checks, and sanitization helpers.
- `review_app.py` is the FastAPI review console.
- `demo.py` runs sample cases from an image directory.
- `worker.py` runs the Temporal worker.
- `tests/` contains activity, utility, and workflow coverage.

## Run Locally

Set `GEMINI_API_KEY` before starting the flow. You can also set `TEMPORAL_ADDRESS`, `MORTGAGE_TASK_QUEUE`, and `UPLOAD_ROOT` if you need overrides.

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

Open `http://localhost:8000` if you are using the review UI. Uploads are written under `UPLOAD_ROOT`, which defaults to `datasets/uploads`.

## Demo And Review Notes

- Upload page images named like `CASEID_p1.png`, `CASEID_p2.png`, and so on.
- The demo runner prefers case-scoped filenames. If a directory contains generic images without case prefixes, it falls back to a single `DEMO-UNSCOPED` workflow.
- The UI starts a workflow when the upload completes successfully.
- Human review is enabled only when the workflow is still running and the current recommendation is `CONDITIONAL`.

## Sample Data

- `datasets/images` contains checked-in sample OCR page images.
- `resources/mortgage_test_cases.json` contains the structured synthetic mortgage fixtures used by tests.

## Run Tests

```bash
uv run poe test -- src/workflows/mortgage_embedded_agent/tests
```

## Compare With

If you want the deterministic baseline first, read [`../mortgage_fixed_flow/README.md`](../mortgage_fixed_flow/README.md).
