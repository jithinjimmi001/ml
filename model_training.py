import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from data_preprocessing import load_and_preprocess

# ---------- Load and split data ----------
X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess()

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,      # use all CPU cores
        verbose=1        # show progress
    )
}

# ---------- Train all models and find the best one ----------
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds)
    print(f"{name} F1 Score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} (F1: {best_score:.4f})")

# Save full model, scaler, and feature order
joblib.dump(best_model, '../backend/model/best_model.pkl')
joblib.dump(scaler, '../backend/model/scaler.pkl')
joblib.dump(feature_names, '../backend/model/feature_names.pkl')

# ---------- Feature Importance (from trained Random Forest) ----------
rf_model = models["RandomForest"]

importances = pd.Series(rf_model.feature_importances_, index=feature_names)
top_features = importances.sort_values(ascending=False).head(5).index.tolist()

print("\nTop 5 most important features:")
print(importances.sort_values(ascending=False).head(5))

# ---------- Build compact feature set: top 5 + Amount (forced in for interpretability) ----------
if 'Amount' not in top_features:
    compact_features = top_features + ['Amount']
else:
    compact_features = top_features

print("\nFinal compact features (with Amount):", compact_features)

X_train_compact = X_train[compact_features]
X_test_compact = X_test[compact_features]

compact_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
compact_model.fit(X_train_compact, y_train)

compact_preds = compact_model.predict(X_test_compact)
compact_f1 = f1_score(y_test, compact_preds)
print(f"Compact model F1 Score (6 features incl. Amount): {compact_f1:.4f}")

# Save compact model + its feature list
joblib.dump(compact_model, '../backend/model/compact_model.pkl')
joblib.dump(compact_features, '../backend/model/compact_features.pkl')

# ---------- Save 5 real GENUINE and 5 real FRAUD examples for demo cycling ----------
genuine_rows = X_test[y_test == 0].head(5)
fraud_rows = X_test[y_test == 1].head(5)

genuine_samples = [row[compact_features].to_dict() for _, row in genuine_rows.iterrows()]
fraud_samples = [row[compact_features].to_dict() for _, row in fraud_rows.iterrows()]

joblib.dump(
    {"genuine": genuine_samples, "fraud": fraud_samples},
    '../backend/model/sample_transactions.pkl'
)
print(f"\nSaved {len(genuine_samples)} genuine and {len(fraud_samples)} fraud sample transactions.")