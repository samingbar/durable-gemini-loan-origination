# Temporal Python SDK Project Template

![GitHub CI](https://github.com/kawofong/temporal-python-template/actions/workflows/ci.yml/badge.svg)
[![Code Coverage](https://img.shields.io/codecov/c/github/kawofong/temporal-python-template.svg?maxAge=86400)](https://codecov.io/github/kawofong/temporal-python-template?branch=master)
[![GitHub License](https://img.shields.io/github/license/kawofong/temporal-python-template)](https://github.com/kawofong/temporal-python-template/blob/main/LICENSE)

## Introduction

A modern, production-ready template for building Temporal applications using [Temporal Python SDK](https://docs.temporal.io/dev-guide/python). This template provides a solid foundation for developing Workflow-based applications with comprehensive testing, linting, and modern Python tooling.

### What's Included

- Complete testing setup (pytest) with async support
- Pre-configured development tooling (e.g. ruff, pre-commit) and CI
- Comprehensive documentation and guides
- [AGENTS.md](https://agents.md/) to provide the context and instructions to help AI coding agents work on your project

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [Temporal CLI](https://docs.temporal.io/cli#install)

### Quick Start

1. **Clone and setup the project:**

   ```bash
   git clone https://github.com/kawofong/temporal-python-template.git
   cd temporal-python-template
   uv sync --dev
   ```

1. **Install development hooks:**

   ```bash
   uv run poe pre-commit-install
   ```

1. **Run tests:**

   ```bash
   uv run poe test
   ```

1. **Start Temporal Server**:

   ```bash
   temporal server start-dev
   ```

1. **Run the example workflow** (in a separate terminal):

   ```bash
   # Start the worker
   uv run -m src.workflows.http.worker

   # In another terminal, execute a workflow
   uv run -m src.workflows.http.http_workflow
   ```

### Mortgage Underwriting Demo (Agentic + Human Review)

This demo runs the Temporal-based mortgage underwriting workflow with Gemini-powered agents and a local human review UI.

1. **Install dependencies:**

   ```bash
   uv sync --dev
   ```

1. **Set environment variables:**

   ```bash
   export GEMINI_API_KEY="your_api_key"
   # Optional: override the Gemini model
   export GEMINI_MODEL="gemini-2.5-flash"
   # Optional: Temporal server address (defaults to localhost:7233)
   export TEMPORAL_ADDRESS="localhost:7233"
   ```

1. **Start a local Temporal server:**

   ```bash
   temporal server start-dev
   ```

1. **Start the worker (new terminal):**

   ```bash
   uv run -m src.workflows.mortgage.worker
   ```

1. **Start the human review UI (new terminal):**

   ```bash
   uv run uvicorn src.workflows.mortgage.review_app:app --reload
   ```

   Open the UI in your browser:

   ```text
   http://localhost:8000
   ```

1. **Run the demo workflow (new terminal):**

   ```bash
   uv run -m src.workflows.mortgage.demo
   ```

   If a case pauses for human review, use the UI to submit a decision.

### Next Steps

- Check out some [example prompts](./docs/example-prompts.md) to generate Temporal Workflows using your favorite tool.
- After you have built your first Temporal Workflow, read [DEVELOPERS.md](./DEVELOPERS.md) to learn about development tips & tricks using this template.
- See [`docs/temporal-patterns.md`](./docs/temporal-patterns.md) for advanced Temporal patterns
- Check [`docs/testing.md`](./docs/testing.md) for Temporal testing best practices

## License

[MIT License](LICENSE).
