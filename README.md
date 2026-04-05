# EE6483 Project: Sentiment Analysis

This repository is structured for the EE6483 assignment on binary sentiment classification of product reviews.

## Project Structure

```text
.
|- configs/
|- data/
|  |- raw/                # train.json, test.json
|  |- processed/          # optional intermediate features
|  \- submissions/        # submission.csv outputs
|- models/                # trained model artifacts
|- notebooks/             # exploratory notebooks
|- reports/               # assignment brief + report drafts
|- scripts/               # runnable train/predict pipeline scripts
\- src/sentiment_project/ # reusable project code
```

## Environment Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional deep-learning stack for BERT/RNN experiments:

```bash
pip install -r requirements-bert.txt
```

## Run Baseline Pipeline

Train TF-IDF + Logistic Regression baseline:

```bash
python scripts/train_tfidf_baseline.py
```

Generate submission from trained model:

```bash
python scripts/generate_submission.py
```

Run train + submission end-to-end in one command:

```bash
python scripts/run_pipeline.py
```

Default outputs:

- Model: `models/tfidf_logreg.joblib`
- Metrics: `reports/train_metrics.json`
- Submission: `data/submissions/submission.csv`

## Model Zoo

Implemented model families:

- Classical: `TF-IDF + LogisticRegression`, `TF-IDF + LinearSVM`, `TF-IDF + MultinomialNB`, `TF-IDF + XGBoost`, `TF-IDF + LightGBM`
- Neural sequence: `TextCNN`, `BiLSTM`, `BiGRU`
- Transformers: `BERT (bert-base-uncased)`, `RoBERTa (roberta-base)`

### Classical benchmark

```bash
python scripts/train_classical_models.py --model-names all --generate-submissions
```

Run a split/seed sweep explicitly:

```bash
python scripts/train_classical_models.py --model-names all --test-sizes 0.2 0.3 0.4 --seeds 42 52 62
```

Each run is saved separately under:

- `models/experiments/<model>/test_size_<ratio>/seed_<seed>/`
- `reports/experiments/<model>/test_size_<ratio>/seed_<seed>/metrics.json`
- `data/submissions/experiments/<model>/test_size_<ratio>/seed_<seed>/submission.csv`

The classical benchmark script also saves:

- raw run ledger: `reports/classical_model_runs.csv`
- aggregated leaderboard: `reports/classical_model_leaderboard.csv`

### Neural models

```bash
python scripts/train_neural_model.py --model-type textcnn
python scripts/train_neural_model.py --model-type bilstm
python scripts/train_neural_model.py --model-type bigru
```

### Transformers

Install deep-learning dependencies first:

```bash
pip install -r requirements-bert.txt
```

Then run:

```bash
python scripts/train_transformer.py --model-name bert-base-uncased
python scripts/train_transformer.py --model-name roberta-base
```

## Comparison Reports

Generate comparison results and charts from available metrics files:

```bash
python scripts/generate_comparison_report.py
```

Comparison outputs are saved under `reports/comparison/`, including:

- `comparison_runs.csv` with one row per `(model_name, test_size, seed)`
- `comparison_results.csv` and `comparison_results.md`
- `metrics_comparison_bar.png`
- per-model confusion matrices in `reports/comparison/confusion_matrices/`
- `confusion_matrix_comparison_grid.png`
- `confusion_pattern_comparison.png`

Each saved metrics file now records:

- `seed`
- `test_size`
- `split_metadata` with train/validation counts and class balance
- `timing` with fit, evaluation, and total runtime

Note: generate metrics first (for example via `train_classical_models.py`, `train_neural_model.py`, `train_transformer.py`) so the comparison script has inputs to aggregate.

## Jupyter

```bash
python -m ipykernel install --user --name ee6483 --display-name "Python (EE6483)"
jupyter notebook
```

## Exploratory Notebooks

- `notebooks/01_exploratory_analysis.ipynb`
  - Full EDA workflow with validation, feature inspection, quick baseline, and `submission_eda_quick.csv`.
- `notebooks/02_tfidf_baseline_experiment.ipynb`
  - TF-IDF hyperparameter sweep, best-config selection, final retrain, and `submission.csv`.
- `notebooks/03_binary_multiclass_sentiment_analysis.ipynb`
  - Binary sentiment training + pseudo 3-class framing (`neg/neu/pos`) via confidence thresholds.
- `notebooks/04_aspect_based_sentiment_analysis.ipynb`
  - Aspect-level sentiment summary using weak supervision.
- `notebooks/05_target_dependent_sentiment_analysis.ipynb`
  - Target-dependent local-context sentiment analysis.
- `notebooks/06_emotion_analysis.ipynb`
  - Emotion-signal analysis and relation to binary polarity.
- `notebooks/07_topic_sentiment_analysis.ipynb`
  - Topic modeling (LDA) combined with sentiment distributions by topic.

## Report Writing

Use `reports/report_template.md` as a checklist for assignment items (a) to (i).
For assignment item (a1), see `reports/literature_survey_foundations.md`.
