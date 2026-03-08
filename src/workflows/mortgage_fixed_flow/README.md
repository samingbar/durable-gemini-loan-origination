# Mortgage Fixed Flow Workflow

This workflow is the deterministic baseline. It runs specialist agents in a fixed order and is easier to reason about when you want consistent behavior.

**When to use this flow**
- You want a stable, repeatable pipeline with fewer moving parts.
- You want the simplest baseline for demos or tests.

**Key files**
- `mortgage_workflow.py` orchestrates the fixed-order pipeline and the human review gate.
- `mortgage_activities.py` handles OCR, policy retrieval, and LLM calls.
- `mortgage_models.py` defines Pydantic models for all inputs and outputs.
- `mortgage_utils.py` contains deterministic metrics, policy checks, and sanitization.
- `review_app.py` is a FastAPI UI for uploads and human review decisions.
- `worker.py` runs the Temporal worker for this workflow.

**Run it locally**

1. Start a Temporal dev server.

```bash
temporal server start-dev
```

1. Run the worker.

```bash
uv run -m src.workflows.mortgage_fixed_flow.worker
```

1. Start the review UI and upload a case.

```bash
uv run uvicorn src.workflows.mortgage_fixed_flow.review_app:app --reload
```

Upload images named like `CASEID_p1.png`, `CASEID_p2.png`, etc. The UI will start the workflow automatically.
