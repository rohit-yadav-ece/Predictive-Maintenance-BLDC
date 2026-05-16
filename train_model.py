"""
Motor Health Classifier — Time-Series Feature Engineering + Neural Network
=============================================================================
Trains a feature-engineered neural network (sklearn MLPClassifier) on
statistical features extracted from raw multi-sensor time-series.

This approach is widely used in industrial predictive maintenance because:
  - Trains 10-50x faster than raw LSTM
  - Achieves comparable accuracy on most fault detection tasks
  - More interpretable feature importance
  - Easier to deploy on edge hardware (ESP32-compatible)

Feature engineering per sensor (5 sensors × 8 features = 40 features):
  mean, std, min, max, range, slope, energy, dominant_freq

Author: Rohit Yadav
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
import joblib

np.random.seed(42)


def extract_features(X):
    """
    Extract 8 statistical features per sensor channel.
    Input:  X of shape (samples, time_steps, sensors)
    Output: features of shape (samples, sensors * 8)
    """
    n_samples, n_steps, n_sensors = X.shape
    features = np.zeros((n_samples, n_sensors * 8))

    for i in range(n_samples):
        for s in range(n_sensors):
            sig = X[i, :, s]
            # Statistical features
            mean_v = sig.mean()
            std_v  = sig.std()
            min_v  = sig.min()
            max_v  = sig.max()
            range_v = max_v - min_v
            # Trend features
            slope = np.polyfit(np.arange(n_steps), sig, 1)[0]
            energy = np.sum(sig ** 2) / n_steps
            # Frequency feature (dominant frequency via FFT)
            fft = np.abs(np.fft.rfft(sig - mean_v))
            dom_freq = np.argmax(fft) if len(fft) > 0 else 0

            base = s * 8
            features[i, base:base+8] = [mean_v, std_v, min_v, max_v,
                                         range_v, slope, energy, dom_freq]
    return features


def main():
    print("=" * 60)
    print("Motor Health Classifier — Feature-Engineered Neural Network")
    print("=" * 60)

    # Load raw time-series
    X = np.load("data/X.npy").astype(np.float32)
    y = np.load("data/y.npy").astype(np.int64)
    print(f"Raw data: {X.shape} (samples, time_steps, sensors)")

    # Extract features
    print("\nExtracting time-series features...")
    X_feat = extract_features(X)
    print(f"Feature matrix: {X_feat.shape} (samples, features)")

    # 70 / 15 / 15 stratified split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_feat, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.1765, random_state=42, stratify=y_tv)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # Train MLP (deep neural network)
    print("\nTraining neural network (3 hidden layers: 64-32-16)...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    y_pred = model.predict(X_test_s)
    y_pred_proba = model.predict_proba(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0)
    label_names = ['Healthy', 'Degrading', 'Critical']

    print(f"\nTraining epochs (iterations): {model.n_iter_}")
    print(f"Total parameters             : {sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_):,}")
    print(f"\nAccuracy : {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall   : {recall*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/mlp_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    metrics = {
        "accuracy":  round(acc * 100, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1_score":  round(f1 * 100, 2),
        "epochs_trained": int(model.n_iter_),
        "test_samples": len(y_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "labels": label_names,
        "model_parameters": int(sum(c.size for c in model.coefs_) +
                                sum(b.size for b in model.intercepts_)),
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training loss curve
    if hasattr(model, 'loss_curve_'):
        pd.DataFrame({
            'epoch': range(1, len(model.loss_curve_) + 1),
            'train_loss': model.loss_curve_,
        }).to_csv("models/training_history.csv", index=False)

    print(f"\n✓ Saved: models/mlp_model.joblib, models/scaler.joblib")
    print(f"✓ Metrics: models/metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
