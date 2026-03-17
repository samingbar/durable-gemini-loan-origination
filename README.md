# Durable Gemini Loan Origination

[![CI](https://github.com/samingbar/durable-gemini-loan-origination/actions/workflows/ci.yml/badge.svg)](https://github.com/samingbar/durable-gemini-loan-origination/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/samingbar/durable-gemini-loan-origination)](LICENSE)

A Temporal-powered set of mortgage underwriting and insurance claims demos that pair Gemini OCR with structured, policy-grounded analysis and human review UIs.

> This is a demo system using synthetic data and simplified policy logic. Do not use it for real lending decisions.

**What you can learn here**
- How to structure deterministic Temporal workflows with non-deterministic activities
- How to ground LLM outputs with policy text
- How to build a human review gate with signals and queries
- How to validate structured LLM output with Pydantic and fallbacks

## Repository Map

- `src/` contains all workflow code, activities, and models.
- `src/workflows/mortgage_fixed_flow/` is the deterministic mortgage baseline pipeline.
- `src/workflows/mortgage_embedded_agent/` adds a supervisor that picks the next specialist to run.
- `src/workflows/insurance_claims_fixed_flow/` is the deterministic insurance claims pipeline.
- `datasets/` contains synthetic inputs and UI uploads.
- `resources/` contains policy PDFs and test cases.
- `docs/` contains Temporal patterns and testing guidance.

If you’re new to Temporal, start with `docs/temporal-primitives.md`.

## Quickstart

### Prerequisites
- Python 3.12+
- `uv` for dependency management
- Temporal CLI (for a local dev server)
- A Gemini API key

### Install

```bash
uv sync --dev
```

### Configure Environment

```bash
# Use either GEMINI_API_KEY or GOOGLE_API_KEY
export GEMINI_API_KEY="your_api_key"
# export GOOGLE_API_KEY="your_api_key"

# Optional: override Gemini model (default: gemini-2.5-flash)
export GEMINI_MODEL="gemini-2.5-flash"

# Optional: Temporal server address (default: localhost:7233)
export TEMPORAL_ADDRESS="localhost:7233"

# Optional: mortgage task queue (default: mortgage-underwriting)
export MORTGAGE_TASK_QUEUE="mortgage-underwriting"

# Optional: insurance task queue (default: insurance-claims)
export INSURANCE_TASK_QUEUE="insurance-claims"

# Optional: upload directory for the mortgage review UI
export UPLOAD_ROOT="datasets/uploads"

# Optional: upload directory for the insurance review UI
export INSURANCE_UPLOAD_ROOT="datasets/uploads/insurance_claims"

# Optional: override the insurance policy corpus path
export INSURANCE_POLICY_PATH="resources/insurance_claim_policies.pdf"
```

### Start Temporal

```bash
temporal server start-dev
```

### Choose a Workflow

**Fixed flow (deterministic baseline)**
- Best when you want repeatable, predictable behavior.
- Specialists always run in the same order.

```bash
uv run -m src.workflows.mortgage_fixed_flow.worker
```

**Insurance claims fixed flow**
- Best when you want the same deterministic OCR+LLM pattern for claim adjudication.
- Coverage, liability, damages, and fraud specialists always run in the same order.

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.worker
```

Generate sample insurance PDFs and OCR images:

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.sample_case_generator
```

**Embedded agent flow (supervisor routing)**
- Best when you want LLM-driven routing and adaptive sequencing.

```bash
uv run -m src.workflows.mortgage_embedded_agent.worker
```

### Run the Human Review UI (Optional)

```bash
# Fixed flow UI
uv run uvicorn src.workflows.mortgage_fixed_flow.review_app:app --reload

# Insurance claims UI
uv run uvicorn src.workflows.insurance_claims_fixed_flow.review_app:app --reload

# Embedded agent UI
uv run uvicorn src.workflows.mortgage_embedded_agent.review_app:app --reload
```

Open `http://localhost:8000` and upload images named like `CASEID_p1.png`, `CASEID_p2.png`, etc.

### Run Sample Cases (Embedded Agent)

```bash
uv run -m src.workflows.mortgage_embedded_agent.demo --image-dir datasets/images
```

If your images are not case-scoped, the demo will process all images as a single case called `DEMO-UNSCOPED`.

### Run Sample Cases (Insurance Claims)

```bash
uv run -m src.workflows.insurance_claims_fixed_flow.demo --image-dir datasets/insurance_claims/images
```

## How It Works (Conceptual Flow)

1. **OCR Intake**: Images are converted to a structured mortgage or insurance payload using Gemini multimodal OCR.
2. **Sanitize + Metrics**: PII is masked and deterministic metrics (DTI/LTV) are computed.
3. **Policy Retrieval**: Relevant policy text is pulled from the workflow's policy corpus.
4. **Specialist Analyses**: Domain-specific specialist analyses are generated in a deterministic order.
5. **Critic Review**: A critic pass checks for missing risks or inconsistencies.
6. **Decision Memo**: LLM drafts a structured decision memo, validated with fallbacks.
7. **Human Review Gate**: Conditional decisions wait for a reviewer signal.

## Data Assets

- `resources/underwriting_policies.pdf` grounds mortgage prompts.
- `resources/insurance_claim_policies.pdf` grounds insurance claim prompts.
- `resources/mortgage_test_cases.json` contains synthetic seed cases.
- `resources/insurance_claim_test_cases.json` contains synthetic insurance claim cases.
- `datasets/images` and `datasets/pdfs` hold generated sample inputs.
- `datasets/insurance_claims/images` and `datasets/insurance_claims/pdfs` hold generated insurance sample inputs.
- `datasets/uploads` is created by the UIs for new uploads (git-ignored).

## Testing & Quality

```bash
# Run tests with coverage
uv run poe test

# Lint and auto-fix
uv run poe lint

# Format code
uv run poe format
```

## Common Tasks

Reset the review UI cache history:

```bash
uv run poe reset-cache
```

Purge uploads as well:

```bash
uv run poe reset-cache -- --purge-uploads
```

## Troubleshooting

- **Workflow won’t start**: Confirm the worker is running and `TEMPORAL_ADDRESS` matches the server.
- **OCR returns invalid JSON**: The workflow has deterministic fallbacks; check logs for the raw response.
- **No cases in UI**: Verify `UPLOAD_ROOT` exists and the UI can write to it.

## License

MIT License. See `LICENSE`.
