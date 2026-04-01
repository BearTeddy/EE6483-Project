# AGENT.md

This repository is organized for reproducible Python data analysis and modeling work, with **Jupyter notebooks as the primary working artifact** for exploration, iteration, and reporting-ready analysis.

## Working Style

- Prefer creating or updating an `.ipynb` notebook first for any data analysis task.
- Use notebooks for:
  - exploratory data analysis
  - feature inspection
  - model comparison
  - error analysis
  - visualization
  - lightweight experiment tracking
- Use Python modules in `src/sentiment_project/` for reusable logic that should not stay embedded in notebooks.
- Use `scripts/` only for CLI entry points and automation wrappers.

## Project Goals

- Keep experiments reproducible.
- Keep raw data immutable.
- Keep outputs organized under `models/`, `reports/`, and `data/submissions/`.
- Keep analysis easy to rerun from top to bottom with minimal manual edits.

## Data Rules

- Read input data from:
  - `data/raw/train.json`
  - `data/raw/test.json`
- Never overwrite files in `data/raw/`.
- Store intermediate cleaned datasets and engineered features in `data/processed/`.
- Any notebook that writes data must write to derived folders only.

## Notebook Rules

- Create notebooks under `notebooks/`.
- Use clear section headers:
  1. Setup
  2. Configuration
  3. Data Loading
  4. Data Validation
  5. Exploratory Analysis
  6. Feature Engineering
  7. Modeling
  8. Evaluation
  9. Inference / Submission
  10. Conclusions / Next Steps
- Make notebooks runnable from top to bottom.
- Use relative paths only.
- Keep outputs meaningful but not noisy.
- Move repeated functions into `src/sentiment_project/`.
- When a notebook becomes stable, convert critical logic into reusable code.

## Reproducibility

- Favor deterministic runs using fixed seeds.
- Default random seed: `42`.
- Record important metrics in `reports/train_metrics.json`.
- Save major artifacts with descriptive names and timestamps only when useful.
- Avoid hidden state between notebook cells.

## Training and Inference Commands

- Baseline training:
  `python scripts/train_tfidf_baseline.py`
- Submission generation:
  `python scripts/generate_submission.py`
- End-to-end pipeline:
  `python scripts/run_pipeline.py`

## Coding Rules

- Put reusable Python code in `src/sentiment_project/`.
- Keep scripts in `scripts/` focused on CLI entry points only.
- Prefer small, testable functions over long notebook-only code blocks.
- Add docstrings for reusable functions.
- Keep configuration near the top of notebooks and scripts.

## Reporting

- Track metrics in `reports/train_metrics.json`.
- Use `reports/report_template.md` to cover assignment items `(a)` to `(i)`.
- Keep figures and tables reproducible from notebook cells where possible.

## Expected Assistant Behavior

When working in this repository, an AI or code assistant should:

1. create or update an `.ipynb` notebook first for analysis work
2. preserve raw data exactly as-is
3. place derived outputs in the correct folders
4. keep notebook code modular and migration-ready
5. surface assumptions clearly
6. default to deterministic settings
7. avoid introducing one-off files outside the project structure

## Preferred First Deliverable for Analysis Tasks

For analysis-heavy tasks, the first deliverable should usually be:

- a notebook in `notebooks/`
- a short summary of what the notebook does
- optional follow-up extraction of reusable code into `src/sentiment_project/`

## Minimal Directory Intent

- `data/raw/` for immutable source data
- `data/processed/` for cleaned or engineered data
- `data/submissions/` for final prediction files
- `models/` for trained artifacts
- `reports/` for metrics and writeups
- `notebooks/` for exploratory and reporting notebooks
- `src/sentiment_project/` for reusable package code
- `scripts/` for command-line entry points
