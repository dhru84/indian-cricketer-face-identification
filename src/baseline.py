"""Classical baseline: Haar cascade crop -> grayscale -> PCA (eigenfaces) -> SVM."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SIZE = 64
_cascade = None


def face_vector(path) -> np.ndarray | None:
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    faces = _cascade.detectMultiScale(img, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = cv2.resize(img[y:y + h, x:x + w], (SIZE, SIZE))
    crop = cv2.equalizeHist(crop)
    return crop.flatten().astype(np.float32) / 255.0


def make_baseline(n_components: int = 50):
    return make_pipeline(StandardScaler(), PCA(n_components=n_components, whiten=True, random_state=0),
                         SVC(kernel="rbf", C=10, gamma="scale"))
