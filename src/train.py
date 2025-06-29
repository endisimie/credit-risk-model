import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from joblib import dump
import os

# Load preprocessed data
X_train = pd.read_csv("../data/processed/models/X_train_processed.csv")
X_test = pd.read_csv("../data/processed/models/X_test_processed.csv")
y_train = pd.read_csv("../data/processed/models/y_train.csv").squeeze()
y_test = pd.read_csv("../data/processed/models/y_test.csv").squeeze()

# Set up MLflow experiment
mlflow.set_experiment("credit-risk-scoring")
#os.makedirs("models", exist_ok=True)
#os.makedirs("figures", exist_ok=True)

# Define models and hyperparameters
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "gbm": GradientBoostingClassifier()
}

params = {
    "logistic_regression": {
        'C': [0.1, 1, 10]
    },
    "gbm": {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
}

best_model = None
best_score = 0

with mlflow.start_run():
    for name, model in models.items():
        print(f"Training and tuning {name}...")

        with mlflow.start_run(nested=True):
            grid = GridSearchCV(model, params[name], cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            best = grid.best_estimator_

            y_pred = best.predict(X_test)
            y_proba = best.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_proba)

            print(f"\nModel: {name}")
            print(f"Accuracy: {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall: {rec:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC: {roc:.4f}")

            # Log metrics and model
            mlflow.log_param("model_name", name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc
            })

            mlflow.sklearn.log_model(best, name)

            # Plot and save Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            cm_path = f"../data/confusion_matrix_{name}.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()

            # Plot and save ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc:.2f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name}")
            plt.legend()
            plt.tight_layout()
            roc_path = f"../data/roc_curve_{name}.png"
            plt.savefig(roc_path)
            mlflow.log_artifact(roc_path)
            plt.close()

            if roc > best_score:
                best_score = roc
                best_model = best
                dump(best_model, "../models/best_model.joblib")
                mlflow.log_artifact("../models/best_model.joblib")

print("✅ Task 5 complete: Models trained, evaluated, visualized, and tracked with MLflow.")
