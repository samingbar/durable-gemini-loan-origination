# Mortgage Embedded Agent Workflow

This workflow demonstrates an agentic loop inside a Temporal workflow. A supervisor activity picks which specialist agent should run next based on progress and risk signals.

**When to use this flow**
- You want a more dynamic, LLM-driven routing strategy.
- You want to compare agentic orchestration against a fixed baseline.

**Key files**
- `mortgage_workflow.py` orchestrates the agentic loop and the human review gate.
- `mortgage_activities.py` handles OCR, policy retrieval, LLM calls, and the supervisor.
- `mortgage_models.py` defines Pydantic models for all inputs and outputs.
- `mortgage_utils.py` contains deterministic metrics, policy checks, and sanitization.
- `review_app.py` is a FastAPI UI for uploads and human review decisions.
- `demo.py` runs sample cases from `datasets/images`.
- `dataset_generator.py` creates synthetic PDFs and images.

**Run it locally**

1. Start a Temporal dev server.

```bash
temporal server start-dev
```

1. Run the worker.

```bash
uv run -m src.workflows.mortgage_embedded_agent.worker
```

1. Start the review UI (optional).

```bash
uv run uvicorn src.workflows.mortgage_embedded_agent.review_app:app --reload
```

1. Run sample cases.

```bash
uv run -m src.workflows.mortgage_embedded_agent.demo --image-dir datasets/images
```

**Generate new synthetic data**

```bash
uv run -m src.workflows.mortgage_embedded_agent.dataset_generator --count 25 --output datasets
```

The generator requires PyMuPDF (`pymupdf`). Install it with `uv add pymupdf` if you plan to run this step.
