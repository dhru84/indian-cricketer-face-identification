"""Train + evaluate all models.   python -m src.train

Expected data layout:  data/raw/<Player_Name>/*.jpg
Outputs: models/face_classifier.joblib and plots/metrics in results/.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import baseline, classifier
from .config import CACHE_DIR, MODEL_PATH, RAW_DIR, RESULTS_DIR, SEED, TEST_SIZE
from .dedupe import dedupe, list_images


def collect():
    """Dedupe images per player and return {player: [paths]}."""
    data = {}
    for folder in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        paths, removed = dedupe(list_images(folder))
        print(f"{folder.name:25s} kept {len(paths):4d}  (removed {removed} near-duplicates)")
        if len(paths) >= 10:
            data[folder.name] = paths
        else:
            print(f"  ! skipping {folder.name}: needs >= 10 usable images")
    return data


def build_embeddings(data):
    from .embeddings import embed_largest_face

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "embeddings.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["paths"]
    X, y, P = [], [], []
    for name, paths in data.items():
        for p in tqdm(paths, desc=name):
            e = embed_largest_face(p)
            if e is not None:
                X.append(e); y.append(name); P.append(str(p))
    X, y, P = np.array(X), np.array(y), np.array(P)
    np.savez(cache, X=X, y=y, paths=P)
    return X, y, P


def plot_confusion(y_true, y_pred, labels, title, out):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, xticks_rotation=45,
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def plot_tsne(X, y, out):
    pts = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=SEED, metric="cosine",
               init="random").fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in np.unique(y):
        m = y == name
        ax.scatter(pts[m, 0], pts[m, 1], s=14, label=name)
    ax.legend(fontsize=7); ax.set_title("t-SNE of FaceNet embeddings (each player forms a cluster)")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    data = collect()
    assert len(data) >= 2, f"Put player folders with images in {RAW_DIR}"
    X, y, P = build_embeddings(data)
    print(f"\n{len(X)} faces from {len(np.unique(y))} players")

    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    results = {}

    # 1) classical baseline (Haar + PCA + SVM)
    vecs = [(baseline.face_vector(P[i]), i) for i in idx]
    ok = {i: v for v, i in vecs if v is not None}
    btr = [i for i in tr if i in ok]; bte = [i for i in te if i in ok]
    base = baseline.make_baseline(min(50, len(btr) - 1)).fit([ok[i] for i in btr], y[btr])
    pred = base.predict([ok[i] for i in bte])
    results["Haar + PCA + SVM (baseline)"] = classifier.scores(y[bte], pred)
    results["Haar + PCA + SVM (baseline)"]["test_faces"] = len(bte)

    # 2) FaceNet embeddings + kNN / SVM
    knn = classifier.make_knn().fit(X[tr], y[tr])
    results["FaceNet + kNN"] = classifier.scores(y[te], knn.predict(X[te]))
    svm = classifier.make_svm().fit(X[tr], y[tr])
    pred = svm.predict(X[te])
    results["FaceNet + SVM (final)"] = classifier.scores(y[te], pred)
    results["FaceNet + SVM (final)"]["test_faces"] = len(te)

    labels = list(svm.classes_)
    plot_confusion(y[te], pred, labels, "FaceNet + SVM confusion matrix", RESULTS_DIR / "confusion_matrix.png")
    report = classification_report(y[te], pred, digits=3)
    (RESULTS_DIR / "classification_report.txt").write_text(report)
    print("\n" + report)

    # 3) open-set: pick the "Unknown" threshold
    rows = classifier.open_set_evaluation(X, y)
    threshold = classifier.best_threshold(rows)
    results["open_set"] = {"chosen_threshold": threshold, "sweep": rows}
    plot_tsne(X, y, RESULTS_DIR / "tsne_embeddings.png")

    # final model: train on everything for deployment
    final = classifier.make_svm().fit(X, y)
    joblib.dump({"model": final, "classes": final.classes_, "threshold": threshold}, MODEL_PATH)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=float))

    print("\nModel comparison")
    for name, r in results.items():
        if name != "open_set":
            print(f"  {name:32s} acc={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}")
    print(f"\nUnknown-person threshold chosen: {threshold}")
    for r in rows:
        print(f"  t={r['threshold']:.1f}  known correct={r['known_correct']:.2f}  unknown rejected={r['unknown_rejected']:.2f}")


if __name__ == "__main__":
    main()
