# EE6483-Project

Project for **EE6483 – Artificial Intelligence & Data Mining**.

---

## Environment Setup

### 1. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment variables (optional)

```bash
cp .env.example .env
# Edit .env and fill in any values you need
```

### 4. Register the kernel and launch Jupyter

```bash
python -m ipykernel install --user --name=ee6483 --display-name "Python (EE6483)"
jupyter notebook
```

---

## Key Libraries

| Library | Purpose |
|---|---|
| `numpy` | Numerical computing |
| `pandas` | Data manipulation & analysis |
| `scipy` | Scientific computing |
| `matplotlib` | Static visualisations |
| `seaborn` | Statistical visualisations |
| `plotly` | Interactive visualisations |
| `scikit-learn` | Machine-learning algorithms |
| `jupyter` / `notebook` | Interactive notebooks |
| `ipykernel` / `ipywidgets` | Jupyter kernel & widgets |
| `python-dotenv` | Load `.env` variables |
