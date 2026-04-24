<!--
Sync Impact Report:
- Version change: none → 1.0.0
- List of modified principles:
  - Added: I. Notebook-First & Simplicity
  - Added: II. Resumable Training
  - Added: III. Sensible Retrieval as the Single Acceptance Gate
  - Added: IV. Retain All Experiment Iterations
- Added sections: Technology Constraints, Acceptance Criteria
- Removed sections: N/A
- Templates requiring updates:
  - .specify/templates/plan-template.md (TODO)
  - .specify/templates/spec-template.md (TODO)
  - .specify/templates/tasks-template.md (TODO)
- Follow-up TODOs: None
-->
# NLP Cross-Modal Food Retrieval System Constitution

## Core Principles

### I. Notebook-First & Simplicity
All core development MUST remain simple and runnable in Jupyter notebooks. There MUST be no premature productionization. Code should be straightforward enough for a university NLP course without operational bloat.

### II. Resumable Training
Training MUST be resumable from checkpoints without losing state. Given the nature of ML experiments, checkpoint mechanisms MUST be robust to prevent lost time.

### III. Sensible Retrieval as the Single Acceptance Gate
The single acceptance gate for success is: does `retrieve(image)` return sensible recipes for a test photo? Complex automated test suites are less important than this core qualitative requirement.

### IV. Retain All Experiment Iterations
Every separate experiment, try, or iteration MUST be saved. This is critical so that performance can be effectively compared across different model architectures or hyperparameter sets later.

## Technology Constraints
Do not add complex infrastructure overhead (Docker swarms, microservices, etc.). Rely purely on Jupyter notebooks and basic Python scripts that are easily executable in standard data science environments.

## Acceptance Criteria
When validating new code or models, the developer MUST run the `retrieve(image)` function with a valid dish photo and subjectively verify that the matched recipes are sensible.

## Governance
This constitution sets the expectations for all course project work.
Amendments should focus on simplifying the codebase and maintaining purely research-focused workflows. 

**Version**: 1.0.0 | **Ratified**: 2026-04-24 | **Last Amended**: 2026-04-24
