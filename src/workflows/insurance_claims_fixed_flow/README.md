# Insurance Claims Fixed Flow Workflow

This workflow is the deterministic baseline for insurance claim adjudication. It mirrors the mortgage fixed flow structure: OCR intake, policy retrieval, fixed-order specialist analyses, critic review, structured decision memo, and a human review gate.

**When to use this flow**
- You want a stable, repeatable claim pipeline with predictable sequencing.
- You want a clean OCR+LLM baseline before adding more agentic routing.

**Key files**
- `insurance_workflow.py` orchestrates the fixed-order pipeline and human review gate.
- `insurance_activities.py` handles OCR, policy retrieval, and LLM calls.
- `insurance_models.py` defines Pydantic models for all inputs and outputs.
- `insurance_utils.py` contains deterministic metrics, policy checks, and sanitization.
- `review_app.py` is a FastAPI UI for uploads and human review decisions.
- `worker.py` runs the Temporal worker for this workflow.

**Run it locally**

1. Start a Temporal dev server.

```bash
temporal server start-dev
```

1. Run the worker.

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.worker
```

1. Start the review UI and upload a case.

```bash
uv run uvicorn src.workflows.insurance_claims_fixed_flow.review_app:app --reload
```

Upload images named like `CASEID_p1.png`, `CASEID_p2.png`, etc. The UI will start the workflow automatically.

**Generate sample OCR cases**

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.sample_case_generator
```

This writes checked-in fixture data to `datasets/insurance_claims/pdfs` and `datasets/insurance_claims/images`.

**Run the sample cases**

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.demo --image-dir datasets/insurance_claims/images
```
