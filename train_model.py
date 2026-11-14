
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "emscad.csv.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "job_fraud_model.joblib"

# Load the dataset
df = pd.read_csv(DATA)

required = {"title", "description", "fraudulent"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Combine available job-related text fields
text_columns = [
    "title", "description", "requirements",
    "company_profile", "benefits"
]
text_columns = [c for c in text_columns if c in df.columns]

df["combined_text"] = (
    df[text_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.strip()
)

df = df[df["combined_text"].ne("")].copy()
df["fraudulent"] = pd.to_numeric(
    df["fraudulent"], errors="coerce"
)
df = df.dropna(subset=["fraudulent"])
df = df[df["fraudulent"].isin([0, 1])]
df["fraudulent"] = df["fraudulent"].astype(int)

X = df["combined_text"]
y = df["fraudulent"]

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError("The dataset needs examples of both classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("features", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=150000,
        sublinear_tf=True,
        min_df=2
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    target_names=["Genuine", "Fraudulent"],
    zero_division=0
))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nSaved trained model to: {MODEL_PATH}")
