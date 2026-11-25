import numpy as np

from src.classifier import best_threshold, make_svm, open_set_evaluation, predict_open_set


def synthetic(n_classes=5, per=40, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, dim)) * 3
    X = np.vstack([c + rng.normal(scale=0.7, size=(per, dim)) for c in centers])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat([f"P{i}" for i in range(n_classes)], per)
    return X, y


def test_svm_separates_clusters():
    X, y = synthetic()
    m = make_svm().fit(X, y)
    labels, conf = predict_open_set(m, X, m.classes_, threshold=0.0)
    assert (labels == y).mean() > 0.95


def test_unknown_rejection_threshold():
    X, y = synthetic()
    keep = y != "P0"
    m = make_svm().fit(X[keep], y[keep])
    lab_low, _ = predict_open_set(m, X[~keep], m.classes_, 0.0)
    lab_high, _ = predict_open_set(m, X[~keep], m.classes_, 0.99)
    assert (lab_high == "Unknown").mean() >= (lab_low == "Unknown").mean()


def test_open_set_evaluation_runs():
    X, y = synthetic()
    rows = open_set_evaluation(X, y)
    assert 0.3 <= best_threshold(rows) <= 0.8
    assert all(0 <= r["balanced"] <= 1 for r in rows)


def test_dedupe_removes_duplicates(tmp_path):
    from PIL import Image

    from src.dedupe import dedupe

    rng = np.random.default_rng(1)
    a = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    a.save(tmp_path / "a.jpg"); a.resize((128, 128)).save(tmp_path / "a_copy.jpg"); b.save(tmp_path / "b.jpg")
    kept, removed = dedupe(sorted(tmp_path.glob("*.jpg")))
    assert removed == 1 and len(kept) == 2
