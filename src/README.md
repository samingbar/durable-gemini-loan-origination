# Source Code

This directory holds the Temporal workflows, activities, models, and utilities that power the demo.

**How the code is organized**
- Workflows live in `src/workflows/<workflow_name>/` and must remain deterministic.
- Activities encapsulate side effects like OCR, PDF parsing, and LLM calls.
- Pydantic models define all inputs and outputs for strong validation.
- Tests live alongside each workflow and use the `_tests.py` naming convention.

**If you are extending the system**
- Keep non-deterministic logic in activities, not workflows.
- Reuse the `mortgage_models.py` schemas to keep data consistent.
- Use the guidance in `docs/write-new-workflow.md` for new workflows.
