# Spec: Temporal + Agent Bricks Are Better Together

## Document Purpose

This document reframes the repository as a thesis-quality reference application for **operational AI**.

The goal is not to present a catalog of unrelated Temporal demos. The goal is to show, with one coherent example and one portability proof, why:

1. **Agent Bricks** is the right place to host and evolve intelligence.
2. **Lakebase** is the right place to persist operational business state.
3. **Temporal** is the right place to run long-lived, failure-tolerant, human-in-the-loop execution.

> Commentary: the existing repository already proves durable orchestration well. What it does not yet prove clearly enough is the platform thesis that intelligence, operational state, and durable execution belong in different layers and become more valuable when combined.

---

## Executive Summary

### Recommendation

Do **not** elevate every workflow package in this repository into an equal first-class thesis example.

That would produce the wrong story shape.

This repository currently contains three runnable workflow packages:

- `mortgage_fixed_flow`
- `mortgage_embedded_agent`
- `insurance_claims_fixed_flow`

Two of those packages are best understood as **two orchestration variants of the same mortgage application**. The third package is best understood as **proof that the same architecture transfers to a second operational domain**.

### Better framing

Use the repository to tell one clear story:

> **Databricks provides the data and intelligence foundation, Lakebase provides the operational system of record, and Temporal provides the durable execution layer that turns model output into reliable real-world outcomes.**

### Thesis structure

Position the repo as:

1. **Canonical reference app:** mortgage processing
2. **A/B comparison inside that app:** deterministic orchestration vs agentic orchestration
3. **Portability proof:** insurance claims

> Commentary: this structure keeps the spotlight on the platform argument. Readers compare orchestration strategies inside one workload, then see the same design pattern reused elsewhere.

---

## Problem Statement

Modern AI systems often succeed at isolated reasoning tasks but fail as production applications because they are missing one or more of these properties:

- durable progress across long time spans
- reliable retries and recovery across infrastructure failures
- explicit human review and override points
- externally visible operational state
- idempotent downstream actions against real systems

The current repository already demonstrates several of these properties through Temporal. The thesis should now make the following stronger claim:

> Intelligence alone is not a production system. Durable execution plus operational state plus human-governed decisioning is what turns model capabilities into a real application.

This is the design claim the code, docs, and demo should prove.

---

## Why This Repository Is A Strong Starting Point

The repository already contains the most important ingredients for a strong operational AI demo:

- OCR and document intake
- policy-grounded retrieval
- multiple specialist analysis steps
- critic or validation passes
- a structured recommendation step
- human review via Temporal signal/query patterns
- long-running workflow execution with pause and resume behavior
- separate worker and review UI surfaces
- synthetic datasets suitable for repeatable demos

### Existing strengths to preserve

#### 1. Human review through workflow state

The review pattern is already one of the strongest parts of the repo:

- workflows publish review packets
- UIs poll query endpoints
- humans submit decisions through signals
- workflows pause and resume durably

This is exactly the kind of behavior that distinguishes a production operational AI application from a simple agent demo.

#### 2. Structured Pydantic contracts

The repo uses typed Pydantic models across activities and workflow boundaries. That is important for the thesis because it makes the line between:

- raw model output
- validated application state
- workflow execution state

both visible and auditable.

#### 3. Deterministic guardrails around model behavior

The repo does not simply wrap model calls. It also governs them. That matters.

Examples already present in the codebase include:

- deterministic fallback logic
- policy checks and hard-stop conditions
- output sanitization
- explicit review gating

These should remain central in the final story because they show Temporal as a control plane, not just a task runner.

#### 4. Insurance review UI maturity

The insurance review app has stronger operational posture than the mortgage UIs:

- explicit case states
- retry support
- degraded readiness behavior
- health and readiness endpoints

That package should be used as evidence that the architecture scales from a demo workflow into something closer to an application surface.

---

## Why “Convert Everything” Is The Wrong Strategy

If every package is presented as a separate thesis pillar, the result will be weaker for four reasons.

### 1. Narrative duplication

The two mortgage packages mostly differ in orchestration style, not business purpose. Presenting them as independent examples creates unnecessary repetition.

### 2. Too much surface area

Readers will spend time learning package boundaries instead of learning the architecture thesis.

### 3. Weak Databricks positioning

The repository is currently strongest on Temporal. If lifted unchanged into a thesis, it will read as a Temporal-centric demo with Databricks mentioned around the edges.

### 4. Demo-catalog smell

A thesis should feel curated. It should read like a reference architecture with deliberate examples, not a set of packages accumulated over time.

> Commentary: the right move is curation, not expansion. The story improves when the number of examples decreases and the architectural clarity increases.

---

## Product Thesis

The target thesis for this repository is:

> Agent Bricks and Temporal are better together because Agent Bricks provides adaptive intelligence, while Temporal provides durable execution, human coordination, and reliable completion semantics for the long-running operational processes that intelligence alone cannot safely finish.

Lakebase completes the picture:

- **Agent Bricks** decides, extracts, routes, summarizes, and critiques
- **Lakebase** stores operational truth used by users and downstream systems
- **Temporal** ensures work completes correctly over time despite retries, delays, failures, and human pauses

### Thesis statement the repository should prove

> Production operational AI requires three separate but coordinated layers:
> intelligence, operational state, and durable execution.

### What “better together” means in practice

Temporal should not replace Agent Bricks.
Agent Bricks should not replace Temporal.

They solve different problems:

| Layer | Primary job | Failure if missing |
| --- | --- | --- |
| Agent Bricks | Generate structured reasoning, extraction, routing, and decision support | Workflow becomes static, brittle, and less adaptive |
| Lakebase | Persist the business-facing state of the case or application | No clear operational record outside workflow history |
| Temporal | Orchestrate long-lived execution with retries, waits, signals, and recovery | AI outputs cannot be turned into reliable multi-step operations |

> Commentary: this table is one of the most important narrative devices in the entire thesis. It makes the division of labor explicit.

---

## Target Story Shape

The repository should be presented as **one reference architecture with two proof styles**.

### Proof style 1: canonical workload

Use mortgage processing as the main application because it already supports:

- deterministic orchestration
- agentic orchestration
- document-centric intake
- policy retrieval
- specialist analyses
- human review

### Proof style 2: portability proof

Use insurance claims to show the architecture is not specific to mortgage underwriting.

This example should be explicitly framed as:

- same architecture
- different ontology
- different policies
- different downstream actions
- same durable execution pattern

### Canonical packaging model

The thesis should conceptually present:

- one **Mortgage Reference App**
- one **Insurance Reference App**

Inside the mortgage reference app, there are two modes:

- `fixed`
- `agentic`

> Commentary: this turns the repository from “three packages” into “one main application with a comparison and one extension.”

---

## Current-State To Target-State Mapping

The repository already contains most of the execution logic needed for the thesis. The main work is to reframe and normalize it.

| Current asset | Current role | Thesis-facing role | Keep | Change |
| --- | --- | --- | --- | --- |
| `src/workflows/mortgage_fixed_flow` | Deterministic mortgage flow | Mortgage reference app, fixed mode | Yes | Normalize docs and shared contracts |
| `src/workflows/mortgage_embedded_agent` | Agentic mortgage flow | Mortgage reference app, agentic mode | Yes | Align more tightly with fixed-mode data model |
| `src/workflows/insurance_claims_fixed_flow` | Deterministic claims flow | Portability proof | Yes | Add explicit platform mapping and external persistence |
| `review_app.py` files | Human review surface | Human operations console | Yes | Position as operational control surface, not just demo UI |
| `sample_case_generator.py` | Fixture generation utility | Synthetic intake artifact generator for repeatable demos | Yes | Document it as part of the end-to-end story |
| Gemini-backed activities | Provider-specific intelligence calls | Agent-provider interface implementation | Partially | Wrap behind Agent Bricks-compatible interfaces |

---

## Reference Architecture

## Architecture Overview

The target architecture should be described as five cooperating layers.

### 1. Intake Layer

Responsibilities:

- receive uploaded documents or page images
- validate case metadata
- stage inputs for workflow execution
- create initial operational case records

Current repo signals:

- review UIs already own file upload handling
- generated insurance PDFs and page images support repeatable intake demos

### 2. Intelligence Layer

Responsibilities:

- OCR and extraction
- context retrieval
- specialist analysis
- supervisor routing
- critic pass
- final recommendation synthesis

Target technology story:

- implemented today through Gemini-backed activities
- abstracted tomorrow as Agent Bricks-compatible tasks

### 3. Durable Execution Layer

Responsibilities:

- orchestrate long-running work
- enforce stage order or agentic routing policy
- handle retries and timeout policies
- wait for human decisions
- resume safely after worker restarts or network interruptions

This is Temporal’s primary domain.

### 4. Operational State Layer

Responsibilities:

- persist case status
- mirror externally meaningful workflow milestones
- store review tasks and assignments
- record structured outputs used by UIs or downstream systems
- support audit and reporting queries

This is Lakebase’s primary domain.

### 5. Downstream Action Layer

Responsibilities:

- update LOS or CRM systems
- create claim tasks
- publish notifications or webhook events
- mark cases as approved, conditional, denied, escalated, or pending

> Commentary: adding this layer is critical. Without a downstream write, the demo ends at “model generated a memo.” A thesis-quality example must continue into operational effect.

---

## Canonical End-To-End Workflow Shape

Every thesis workflow should fit the same high-level sequence.

### 1. Case intake

- create case in Lakebase
- store intake metadata
- bind uploaded documents to case and workflow run

### 2. Document extraction

- invoke OCR or structured extraction through the intelligence interface
- persist extracted document metadata

### 3. Policy and context retrieval

- fetch relevant policy or business rules
- persist the retrieval trace or references used for explainability

### 4. Specialist analysis

- run either fixed-order specialists or agentically selected specialists
- persist structured outputs per specialist

### 5. Critic and validation

- detect contradictions, missing evidence, unsupported conclusions, or policy conflicts
- determine whether a human review gate is needed

### 6. Recommendation synthesis

- produce a structured recommendation
- record rationale and confidence

### 7. Human review gate

- publish review packet via query or update-safe workflow state
- mirror review task into Lakebase
- await reviewer action through signal or update

### 8. Operational persistence

- write final decision or pending-review status into Lakebase
- mark workflow phase progression explicitly

### 9. Downstream action

- execute idempotent downstream write activity
- record external system identifiers and sync status

### 10. Audit completion

- emit final audit event
- mark workflow run complete
- preserve decision packet for later review

> Commentary: this canonical sequence is what makes the repository look like an operational application instead of a workflow demo.

---

## Detailed Division Of Labor

## What Agent Bricks Should Own

The intelligence layer should be expressed through a provider-agnostic contract with implementations that can be backed by Gemini today and Agent Bricks tomorrow.

Suggested responsibilities:

- document extraction
- structured field inference
- retrieval-augmented specialist reasoning
- agentic routing or supervisor decisions
- critic pass and consistency checks
- recommendation memo drafting

Suggested interface modules:

```text
src/platform/agents/
  interface.py
  gemini_provider.py
  agent_bricks_provider.py
```

### Agent contracts to define

The spec should explicitly define input/output contracts for:

1. extraction
2. policy retrieval synthesis
3. specialist analysis
4. supervisor routing
5. critic evaluation
6. final decision recommendation

Each contract should require:

- a typed request payload
- a typed response payload
- traceable provenance fields
- explicit failure and fallback semantics

### Why this abstraction matters

Without this layer, the repo reads as “Temporal + Gemini.”
With this layer, it reads as “Temporal + pluggable intelligence, where Agent Bricks is the intended production implementation.”

---

## What Temporal Should Own

Temporal should remain responsible for orchestration, not business-state storage and not raw model reasoning.

Temporal responsibilities should be documented as:

- sequencing work
- coordinating parallel activities
- applying retry and timeout policies
- pausing on human input
- recovering from worker restarts
- exposing review packet state
- ensuring idempotent downstream step completion

### Workflow design principles to preserve

1. Workflows remain deterministic.
2. Activities own side effects and non-deterministic operations.
3. Human review remains explicit and durable.
4. Model outputs are validated before influencing irreversible actions.
5. External writes are idempotent and replay-safe.

### Why Temporal matters in the thesis

Agentic reasoning is useful but insufficient for production operations.
The workflow engine is what makes it safe to:

- wait hours or days for humans
- retry after transient provider failures
- survive process restarts
- observe progress across a case lifecycle
- complete downstream actions exactly once from the application point of view

---

## What Lakebase Should Own

Lakebase should be introduced as the operational database of record for application-facing state.

### Critical rule

**Temporal workflow history is not the operational system of record.**

Temporal stores durable execution history.
Lakebase stores business-facing state consumed by:

- review UIs
- operations teams
- external systems
- audit and reporting tools

### Proposed Lakebase schema

Suggested core tables:

- `cases`
- `documents`
- `workflow_runs`
- `analysis_results`
- `review_tasks`
- `audit_events`
- `downstream_actions`

### Suggested table responsibilities

#### `cases`

- business-facing case identity
- current status
- product or domain type
- applicant or claimant metadata
- current decision summary

#### `documents`

- uploaded artifact metadata
- source URI or storage key
- extraction status
- page count, hash, and provenance

#### `workflow_runs`

- workflow ID
- run ID
- orchestration mode
- current stage
- start and completion timestamps
- execution status

#### `analysis_results`

- specialist name
- structured result payload
- confidence or risk signals
- model trace or provider metadata

#### `review_tasks`

- current review queue item
- assignee
- SLA target
- review status
- reviewer submission payload

#### `audit_events`

- timestamped event stream
- workflow milestone transitions
- external action outcomes
- human review actions

#### `downstream_actions`

- external target system
- idempotency key
- action payload summary
- sync status
- external record ID

### Repository module shape

```text
src/platform/lakebase/
  models.py
  repository.py
  migrations/
```

### Lakebase write points

The workflows should write to Lakebase at these points:

1. case creation
2. document registration
3. extraction completion
4. specialist result completion
5. review task creation
6. reviewer submission
7. final recommendation
8. downstream action success or failure
9. workflow completion

> Commentary: these write points are where the thesis becomes concrete. They are also where operational visibility is won or lost.

---

## Human Review Lifecycle

Human review is not an edge case. It is a core proof point for why Temporal is required.

### Lifecycle model

1. workflow reaches a reviewable state
2. workflow exposes a review packet
3. workflow creates or updates a `review_tasks` row in Lakebase
4. UI renders the review packet and current case context
5. reviewer submits approve, deny, conditionally approve, or request changes
6. workflow resumes through a signal or update
7. workflow persists the review outcome
8. workflow executes downstream actions

### What the current repo already proves

- review packet query patterns
- durable wait semantics
- signal-driven resume
- richer case lifecycle UI behavior in the insurance flow

### What the thesis should add

- mirrored review state in Lakebase
- explicit review assignment model
- audit trail entries for every human action
- SLA and retry semantics for unreviewed cases

---

## Downstream Actions

At least one workflow must perform an idempotent external write after final decision or review completion.

### Why this matters

A system that stops at “generated recommendation” is still a reasoning demo.
A system that completes a downstream action is an operational application.

### Example downstream actions

For mortgage:

- create underwriting task in LOS
- write approval conditions to application record
- open fraud escalation
- notify loan processor

For insurance:

- update claim status in claims system
- create special investigations unit task
- trigger payment hold
- notify adjuster queue

### Adapter modules

```text
src/platform/downstream/
  los_adapter.py
  claims_adapter.py
  notifications.py
```

### Required behavior

- idempotency key per action
- retriable transient failure handling
- persisted sync status in Lakebase
- auditable final state

---

## Recommended Repository Framing

The docs should stop leading with “choose a workflow.”
They should start with the architecture thesis.

### New top-level README outline

1. architecture thesis
2. why Agent Bricks + Lakebase + Temporal fit together
3. reference architecture diagram
4. mortgage reference app
5. fixed vs agentic comparison
6. insurance portability proof
7. local run instructions
8. demo walkthrough

### Recommended conceptual naming

Without necessarily renaming the Git repository immediately, the documentation should conceptually position the project as one of the following:

- `Operational AI Reference App`
- `Databricks + Temporal Operational AI Reference`
- `Durable Operational AI on Databricks`

> Commentary: renaming documentation before renaming code is the lowest-risk way to shift the story quickly.

---

## Mortgage Reference App

Mortgage should be the canonical workload because it supports the clearest orchestration comparison.

### Fixed mode

Backed by `mortgage_fixed_flow`.

Positioning:

- deterministic stage order
- easiest mode to reason about
- strongest baseline for regression testing
- preferred when policy and process are stable

### Agentic mode

Backed by `mortgage_embedded_agent`.

Positioning:

- supervisor-selected specialist sequence
- better for variable case complexity
- requires stronger observability and fallback design
- useful to show why durable execution is even more important for adaptive systems

### A/B comparison goal

The thesis should make it obvious that the two modes differ mainly in **orchestration policy**, not in:

- domain model
- review semantics
- operational persistence model
- downstream effect model

### Refactor target

The end-state should converge on a shared mortgage core:

```text
src/reference_app/mortgage/
  common/
    models.py
    activities.py
    review_models.py
    utils.py
  fixed/
    workflow.py
    worker.py
  agentic/
    workflow.py
    worker.py
  review_app.py
```

This is not required for phase 1, but it should be the target shape.

---

## Insurance As Portability Proof

Insurance should remain in the repository, but its role should be narrow and deliberate.

### Insurance thesis role

Insurance proves:

- the architecture is not mortgage-specific
- the review pattern generalizes
- policy-grounded decisioning generalizes
- operational persistence generalizes
- downstream action semantics generalize

### What to preserve

- fixed-order adjudication structure
- richer review UI behavior
- retry and degraded-mode readiness posture
- synthetic fixture generation utilities

### What to add

- explicit Lakebase persistence points
- at least one downstream claims-system action
- docs that map the claims flow onto the same architecture diagram used for mortgage

### Why the insurance utility matters

`src/workflows/insurance_claims_fixed_flow/utils/sample_case_generator.py` should be described as more than a helper script. It is the asset generator that makes the insurance demo:

- reproducible
- explainable
- easy to stage live

That is useful in a thesis context because repeatable demos are often more persuasive than abstract design claims.

---

## Implementation Strategy

## Phase 1: Reframe Without Destabilizing

### Goal

Make the repo thesis-ready through documentation and light platform seams while leaving existing workflow logic largely intact.

### Deliverables

1. rewritten top-level README
2. architecture doc
3. deterministic-vs-agentic comparison doc
4. Lakebase integration doc
5. Agent Bricks integration doc
6. target package map from current code to thesis architecture

### Minimal code changes

- add persistence interfaces
- add provider interfaces
- add downstream adapter stubs
- add structured event logging where needed

### Explicit non-goal for phase 1

Do not rewrite working workflow packages just to make the tree look cleaner.

> Commentary: phase 1 is about narrative truthfulness and integration seams, not cosmetic refactors.

---

## Phase 2: Normalize Mortgage Into One Comparison

### Goal

Turn the two mortgage packages into one coherent comparison with a shared domain core.

### Work items

- align Pydantic models
- align review packet structures
- align persistence write points
- extract shared utilities
- isolate orchestration differences into separate workflows

### End-state claim

By the end of phase 2, the mortgage example should support the statement:

> Same application, same business outcome, same operational model, different orchestration strategy.

That is the cleanest possible A/B story for a thesis.

---

## Phase 3: Promote Insurance As Cross-Domain Evidence

### Goal

Show that the design pattern applies beyond mortgage.

### Work items

- align insurance docs with the shared platform architecture
- persist insurance case and review state to Lakebase
- add one downstream claims-system action
- preserve the stronger review UI as an operational UX exemplar

### End-state claim

Insurance should answer the reader’s likely objection:

> “Is this just a mortgage demo?”

The answer should be no, and the evidence should be concrete.

---

## Proposed Artifact Set

The following artifacts should be produced from the conversion effort.

### 1. Architecture spec

Should include:

- system diagram
- sequence diagram
- layer responsibilities
- state model
- failure model
- review lifecycle

### 2. Mortgage comparison doc

Should include:

- fixed vs agentic behavior
- use-case fit
- reliability tradeoffs
- observability differences
- control mechanisms

### 3. Lakebase integration spec

Should include:

- schema
- write points
- idempotency keys
- transaction boundaries
- read models
- audit behavior

### 4. Agent Bricks integration spec

Should include:

- provider contracts
- fallback semantics
- output validation rules
- traceability model

### 5. Demo walkthrough

Should show:

- intake
- analysis
- review pause
- human decision
- resume
- downstream write
- operational state visibility

---

## Demo Narrative

The live demo matters because it proves the thesis in one pass.

### Recommended demo sequence

1. Upload a mortgage case
2. Start the worker and show workflow execution beginning
3. Show policy retrieval and specialist analysis
4. Surface a case that requires human review
5. Pause durably while the UI displays the review packet
6. Submit a reviewer decision
7. Resume the workflow
8. Persist the final decision to Lakebase
9. Execute a downstream action
10. Show the final audit trail

### Why this sequence works

It highlights the exact boundary between:

- model intelligence
- workflow orchestration
- human intervention
- operational persistence
- real-world completion

### Optional second demo

Run the insurance flow immediately after the mortgage flow to show the same architecture pattern with:

- different domain entities
- different policy corpus
- different review semantics
- different downstream action

This demonstrates portability without opening a second architecture discussion.

---

## Design Rules

These rules should govern all thesis-facing changes.

### Rule 1

Workflows remain deterministic.

### Rule 2

All side effects stay in activities or adapter layers.

### Rule 3

Operational business state lives in Lakebase, not only in workflow history.

### Rule 4

Provider-specific model calls are hidden behind stable interfaces.

### Rule 5

Every externally meaningful state change should be auditable.

### Rule 6

Every irreversible downstream action should be idempotent.

### Rule 7

Human review should be modeled as a first-class lifecycle, not an exception path.

### Rule 8

The mortgage example is canonical. Insurance is corroborating evidence.

---

## Non-Goals

This spec should explicitly rule out a few tempting but distracting directions.

### Non-goal 1

Do not turn every package into a separate product story.

### Non-goal 2

Do not collapse Lakebase and Temporal into a single state concept.

### Non-goal 3

Do not let provider-specific Gemini implementation details dominate the narrative.

### Non-goal 4

Do not pursue a large code reorganization before the documentation and platform seams are clear.

### Non-goal 5

Do not present the architecture as “fully autonomous AI.”
The value proposition is governed operational AI.

---

## Risks And Mitigations

### Risk: the repo still reads as Temporal-first, Databricks-second

Mitigation:

- add explicit Agent Bricks and Lakebase layers in docs
- introduce provider and persistence interfaces in code
- include Lakebase write points in sequence diagrams

### Risk: the mortgage variants drift too far apart

Mitigation:

- align data contracts first
- extract shared review models
- compare orchestration policy only

### Risk: the demo ends at recommendation generation

Mitigation:

- require at least one downstream action in the canonical demo
- persist external sync status visibly

### Risk: insurance feels bolted on

Mitigation:

- use the exact same architecture vocabulary and sequence shape
- position insurance only as portability proof

---

## Acceptance Criteria

The conversion is successful when all of the following are true.

### Narrative criteria

- a reader can explain the role of Agent Bricks, Lakebase, and Temporal in one pass
- the repository reads like a reference architecture, not a bag of demos
- mortgage is clearly the canonical workload
- insurance clearly plays the role of portability proof

### Technical criteria

- every workflow persists externally meaningful state to Lakebase
- human review state is visible both in workflow state and database state
- at least one workflow performs an idempotent downstream write
- mortgage fixed and agentic modes share the same core data model
- orchestration strategy is the main difference between the mortgage modes

### Demo criteria

- one end-to-end demo runs from intake to review to downstream action
- deterministic and agentic mortgage modes can be compared directly
- insurance proves architecture portability instead of adding noise

---

## Explicit Pushback

If the ask is interpreted literally as:

> “Convert every Temporal workflow in this repository into a separate thesis example.”

The answer should be:

**No. Curate them.**

The stronger move is:

- one canonical mortgage reference app
- one fixed vs agentic comparison inside that app
- one insurance portability proof

That is the structure most likely to make the thesis memorable and credible.

---

## Recommended Immediate Next Steps

1. Rewrite the top-level README around the architecture thesis.
2. Add a new architecture document with a system diagram and sequence diagram.
3. Add provider and persistence interfaces without rewriting workflow logic.
4. Document the mortgage flows as two modes of one reference application.
5. Document insurance as the cross-domain portability example.
6. Add one explicit downstream action to the canonical demo path.

---

## Appendix A: Existing Repo Assets Worth Highlighting

### Mortgage fixed flow

Best used to show:

- deterministic orchestration
- predictable stage ordering
- straightforward review gating
- stable baseline behavior

### Mortgage embedded agent flow

Best used to show:

- adaptive supervisor routing
- the need for fallback protection
- the value of Temporal when execution order is not fully static

### Insurance fixed flow

Best used to show:

- stronger operations-facing UI behavior
- portability of the architecture to another domain
- repeatable demo assets and review states

### Insurance sample case generator

Best used to show:

- repeatable synthetic document generation
- artifact preparation for OCR demos
- a thesis-friendly story around reproducible operational AI demos

---

## Appendix B: Suggested Target Package Layout

```text
src/
  platform/
    agents/
      interface.py
      gemini_provider.py
      agent_bricks_provider.py
    lakebase/
      models.py
      repository.py
      migrations/
    downstream/
      los_adapter.py
      claims_adapter.py
      notifications.py
  reference_app/
    mortgage/
      common/
        models.py
        activities.py
        review_models.py
        utils.py
      fixed/
        workflow.py
        worker.py
      agentic/
        workflow.py
        worker.py
      review_app.py
    insurance/
      common/
        models.py
        activities.py
        review_models.py
        utils.py
      fixed/
        workflow.py
        worker.py
      review_app.py
docs/
  architecture.md
  mortgage-fixed-vs-agentic.md
  lakebase-integration.md
  agent-bricks-integration.md
  demo-walkthrough.md
  temporal_databricks_thesis_spec.md
```

This layout is a target, not a phase-1 requirement.

---

## Final Position

The winning thesis is not:

> “Here are several Temporal demos.”

The winning thesis is:

> “Here is a reference architecture for operational AI in which Agent Bricks supplies adaptive intelligence, Lakebase supplies operational truth, and Temporal supplies the durable execution needed to reliably turn decisions into outcomes.”

That is the story this repository is already close to proving. The work now is to make that story explicit, disciplined, and easy to demonstrate.
