"""
Realistic BLDC Motor Fault Data Generator
==========================================
Generates synthetic time-series sensor data simulating real motor degradation:
- Healthy state: Normal operation patterns
- Degrading state: Gradual drift in vibration, temperature, current
- Critical state: Severe anomalies preceding failure

Sensors simulated:
  - Vibration RMS (g)
  - Bearing temperature (°C)
  - Stator current (A)
  - Acoustic emission (dB)
  - Rotational speed (RPM)

Author: Rohit Yadav
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
np.random.seed(42)

# Configuration
SAMPLES_PER_CLASS = 1500       # Time series per class
SEQUENCE_LENGTH = 50            # Time steps per sequence
SENSORS = ['vibration', 'temperature', 'current', 'acoustic', 'rpm']


def generate_healthy_sequence():
    """Normal operation — stable readings with small Gaussian noise."""
    vibration   = np.random.normal(0.5, 0.08, SEQUENCE_LENGTH)
    temperature = np.random.normal(45,  2.0,  SEQUENCE_LENGTH)
    current     = np.random.normal(2.5, 0.15, SEQUENCE_LENGTH)
    acoustic    = np.random.normal(55,  2.5,  SEQUENCE_LENGTH)
    rpm         = np.random.normal(1500, 15,  SEQUENCE_LENGTH)
    return np.stack([vibration, temperature, current, acoustic, rpm], axis=1)


def generate_degrading_sequence():
    """Early bearing/winding degradation — slow drift + intermittent spikes."""
    t = np.linspace(0, 1, SEQUENCE_LENGTH)
    drift = t * np.random.uniform(0.8, 1.2)

    vibration   = np.random.normal(0.5, 0.08, SEQUENCE_LENGTH) + drift * 0.6
    temperature = np.random.normal(45,  2.0,  SEQUENCE_LENGTH) + drift * 12
    current     = np.random.normal(2.5, 0.15, SEQUENCE_LENGTH) + drift * 0.4
    acoustic    = np.random.normal(55,  2.5,  SEQUENCE_LENGTH) + drift * 7
    rpm         = np.random.normal(1500, 15,  SEQUENCE_LENGTH) - drift * 30

    # Inject intermittent spikes (early fault signature)
    spike_idx = np.random.choice(SEQUENCE_LENGTH, size=3, replace=False)
    vibration[spike_idx] += np.random.uniform(0.2, 0.4, 3)
    acoustic[spike_idx]  += np.random.uniform(4, 8, 3)

    return np.stack([vibration, temperature, current, acoustic, rpm], axis=1)


def generate_critical_sequence():
    """Severe fault — high amplitude, irregular patterns, imminent failure."""
    t = np.linspace(0, 1, SEQUENCE_LENGTH)
    severity = t * np.random.uniform(1.5, 2.2)

    vibration   = np.random.normal(0.5, 0.20, SEQUENCE_LENGTH) + severity * 1.5
    temperature = np.random.normal(45,  4.0,  SEQUENCE_LENGTH) + severity * 25
    current     = np.random.normal(2.5, 0.40, SEQUENCE_LENGTH) + severity * 1.2
    acoustic    = np.random.normal(55,  5.0,  SEQUENCE_LENGTH) + severity * 18
    rpm         = np.random.normal(1500, 40,  SEQUENCE_LENGTH) - severity * 120

    # Multiple severe spikes
    n_spikes = np.random.randint(5, 9)
    spike_idx = np.random.choice(SEQUENCE_LENGTH, size=n_spikes, replace=False)
    vibration[spike_idx] += np.random.uniform(0.5, 1.0, n_spikes)
    current[spike_idx]   += np.random.uniform(0.8, 1.5, n_spikes)
    acoustic[spike_idx]  += np.random.uniform(10, 20, n_spikes)

    return np.stack([vibration, temperature, current, acoustic, rpm], axis=1)


def build_dataset():
    """Generate the complete dataset."""
    X, y = [], []
    label_map = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}
    generators = [generate_healthy_sequence, generate_degrading_sequence, generate_critical_sequence]

    for label, gen in enumerate(generators):
        for _ in range(SAMPLES_PER_CLASS):
            X.append(gen())
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    return X, y, label_map


if __name__ == "__main__":
    print("=" * 60)
    print("BLDC Motor Fault Data Generator")
    print("=" * 60)

    X, y, label_map = build_dataset()

    print(f"\nDataset shape  : {X.shape}  (samples, time_steps, sensors)")
    print(f"Labels shape   : {y.shape}")
    print(f"Total samples  : {len(X)}")
    print(f"Sensors        : {SENSORS}")
    print(f"\nClass distribution:")
    for label, name in label_map.items():
        count = (y == label).sum()
        print(f"  {name:10s} : {count}")

    # Save dataset
    os.makedirs("data", exist_ok=True)
    np.save("data/X.npy", X)
    np.save("data/y.npy", y)

    # Also save a CSV preview of first sequence per class
    preview_rows = []
    seen_classes = set()
    for i in range(len(X)):
        if y[i] not in seen_classes:
            seq = X[i]
            for t in range(min(10, SEQUENCE_LENGTH)):
                preview_rows.append({
                    'class': label_map[y[i]],
                    'time_step': t,
                    'vibration': seq[t, 0],
                    'temperature': seq[t, 1],
                    'current': seq[t, 2],
                    'acoustic': seq[t, 3],
                    'rpm': seq[t, 4],
                })
            seen_classes.add(y[i])
            if len(seen_classes) == 3:
                break
    pd.DataFrame(preview_rows).to_csv("data/sample_preview.csv", index=False)

    print(f"\n✓ Saved: data/X.npy, data/y.npy, data/sample_preview.csv")
    print("=" * 60)
