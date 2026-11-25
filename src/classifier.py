"""Embedding classifiers and open-set ("Unknown person") logic. Pure sklearn/numpy,
so it can be unit-tested without a GPU or a dataset."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def make_svm():
    # Calibrated linear SVM -> trustworthy probabilities for the Unknown threshold.
    return CalibratedClassifierCV(SVC(kernel="linear", C=1.0), method="sigmoid", cv=3, ensemble=False)


def make_knn(k: int = 5):
    return KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")


def predict_open_set(model, X: np.ndarray, classes: np.ndarray, threshold: float, unknown="Unknown"):
    """Return (label, confidence) per row; label=Unknown if max prob < threshold."""
    proba = model.predict_proba(X)
    idx = proba.argmax(1)
    conf = proba.max(1)
    labels = np.where(conf >= threshold, classes[idx], unknown)
    return labels, conf


def open_set_evaluation(X: np.ndarray, y: np.ndarray, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                        seed: int = 42) -> list[dict]:
    """Leave-one-class-out simulation of strangers.

    For each player P: train on the other players (P is truly unseen), then test on
    held-out images of the remaining players (should be accepted + correct) and on
    P's images (should be rejected as Unknown).
    Returns per-threshold averages of known-accept rate and unknown-reject rate.
    """
    classes = np.unique(y)
    acc = {t: {"known_correct": [], "unknown_rejected": []} for t in thresholds}
    for held_out in classes:
        known_mask = y != held_out
        Xk, yk = X[known_mask], y[known_mask]
        Xtr, Xte, ytr, yte = train_test_split(Xk, yk, test_size=0.25, stratify=yk, random_state=seed)
        model = make_svm().fit(Xtr, ytr)
        Xu = X[~known_mask]
        for t in thresholds:
            lab_k, _ = predict_open_set(model, Xte, model.classes_, t)
            lab_u, _ = predict_open_set(model, Xu, model.classes_, t)
            acc[t]["known_correct"].append(float(np.mean(lab_k == yte)))
            acc[t]["unknown_rejected"].append(float(np.mean(lab_u == "Unknown")))
    rows = []
    for t in thresholds:
        kc = float(np.mean(acc[t]["known_correct"]))
        ur = float(np.mean(acc[t]["unknown_rejected"]))
        rows.append({"threshold": t, "known_correct": kc, "unknown_rejected": ur,
                     "balanced": (kc + ur) / 2})
    return rows


def best_threshold(rows: list[dict]) -> float:
    return max(rows, key=lambda r: r["balanced"])["threshold"]


def scores(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}
