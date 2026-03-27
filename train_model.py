import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("🧠 Training Multi-Output Pest Risk Model...")

# ── Load dataset ──
csv_path = os.path.join(os.path.dirname(__file__), "pest_dataset.csv")
df = pd.read_csv(csv_path)
print(f"📂 Loaded {len(df)} rows from {csv_path}")

# ── Features & Labels ──
feature_cols = ["ndvi", "temperature", "humidity", "rainfall"]
X = df[feature_cols].values

# Encode categorical labels
le_risk = LabelEncoder()
le_pest = LabelEncoder()

y_risk = le_risk.fit_transform(df["risk_level"])
y_pest = le_pest.fit_transform(df["pest_type"])

y = np.column_stack([y_risk, y_pest])

# ── Train / Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_risk
)

# ── Train Multi-Output Random Forest ──
base_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model = MultiOutputClassifier(base_rf)
model.fit(X_train, y_train)

# ── Evaluate ──
y_pred = model.predict(X_test)

print("\n" + "=" * 50)
print("📈 RISK LEVEL RESULTS")
print("=" * 50)
risk_acc = accuracy_score(y_test[:, 0], y_pred[:, 0])
print(f"Accuracy: {risk_acc:.2%}")
print(classification_report(
    y_test[:, 0], y_pred[:, 0],
    target_names=le_risk.classes_
))

print("=" * 50)
print("🐛 PEST TYPE RESULTS")
print("=" * 50)
pest_acc = accuracy_score(y_test[:, 1], y_pred[:, 1])
print(f"Accuracy: {pest_acc:.2%}")
print(classification_report(
    y_test[:, 1], y_pred[:, 1],
    target_names=le_pest.classes_
))

# ── Save model & encoders ──
base_dir = os.path.dirname(__file__)

model_path = os.path.join(base_dir, "pest_model.pkl")
joblib.dump(model, model_path)
print(f"\n💾 Model saved → {model_path}")

encoders = {"risk_level": le_risk, "pest_type": le_pest}
enc_path = os.path.join(base_dir, "label_encoders.pkl")
joblib.dump(encoders, enc_path)
print(f"💾 Label encoders saved → {enc_path}")

print("\n✅ Training complete!")
