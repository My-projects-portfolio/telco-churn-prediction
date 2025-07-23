import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_and_clean_data


# Create results folder
os.makedirs("results", exist_ok=True)

def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("results/confusion_matrix.png")
    plt.close()

def plot_roc(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("results/roc_curve.png")
    plt.close()

def plot_feature_importance(model, X):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10
    features = X.columns[indices]

    plt.figure(figsize=(8,6))
    plt.title("Top 10 Important Features")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), features)
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png")
    plt.close()

def train_model(data_path):
    # Load and preprocess data
    df = load_and_clean_data(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=False)
    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    print("Model trained successfully. Evaluation results saved in 'results/'")

    # Save model
    joblib.dump(model, 'models/churn_model.pkl')

    # Generate Plots
    plot_confusion(y_test, y_pred)
    plot_roc(y_test, y_proba)
    plot_feature_importance(model, X)

if __name__ == "__main__":
    train_model("data/telecom_churn.csv")
