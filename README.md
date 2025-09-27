# Fake News Detector

Binary text classifier to distinguish **FAKE** vs **REAL** news using TF-IDF features and classic ML models. Includes EDA, baseline Logistic Regression, Linear SVM, and XGBoost with saved metrics, confusion matrix, and model artifacts.

## Data
Place the Kaggle-style CSVs in the project root:
- `Fake.csv`
- `True.csv`

The script will create:
- `combined_news_data.csv` (merged dataset)
- `confusion_matrix.png`
- `model_metrics.csv`
- `model_comparison.csv`
- `fake_news_model.pkl`, `tfidf_vectorizer.pkl`

## Methods
- **EDA:** label distribution, missing values, basic stats, title/text length histograms.
- **Features:** `TfidfVectorizer` on `title + text`.
- **Models:**
  - Logistic Regression
  - Linear SVM (`LinearSVC`)
  - XGBoost (`XGBClassifier`)
- **Metrics:** Accuracy, Precision, Recall, F1; confusion matrix heatmap.
- **Model comparison:** writes a CSV with all metrics.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Usage
```bash
python load_data.py
```

This will:
- Load and merge `Fake.csv` and `True.csv`
- Train TF-IDF + Logistic Regression (baseline)
- Train Linear SVM and XGBoost for comparison
- Print metrics and save artifacts/plots

## Files produced
- `confusion_matrix.png` — confusion matrix heatmap
- `model_metrics.csv` — metrics for the baseline run
- `model_comparison.csv` — LR vs SVM vs XGBoost
- `fake_news_model.pkl`, `tfidf_vectorizer.pkl` — saved model + vectorizer

## Next steps
- Hyperparameter tuning (GridSearchCV / Optuna)
- Class-weighting or calibrated probabilities for SVM
- Add validation curves and learning curves
- Optional: Streamlit demo for interactive predictions
