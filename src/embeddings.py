"""Face detection (MTCNN) + 512-d embeddings (FaceNet / InceptionResnetV1)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_device = "cuda" if torch.cuda.is_available() else "cpu"
_mtcnn = None
_resnet = None


def _models():
    global _mtcnn, _resnet
    if _mtcnn is None:
        from facenet_pytorch import MTCNN, InceptionResnetV1

        _mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, post_process=True, device=_device)
        _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
    return _mtcnn, _resnet


@torch.no_grad()
def detect_and_embed(img: Image.Image, min_prob: float = 0.90):
    """Return a list of dicts: {box, prob, embedding} for every face found."""
    mtcnn, resnet = _models()
    img = img.convert("RGB")
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return []
    faces = mtcnn(img)  # tensor [n,3,160,160] aligned crops (same order as detect)
    if faces is None:
        return []
    if faces.ndim == 3:
        faces = faces.unsqueeze(0)
    emb = resnet(faces.to(_device)).cpu().numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    out = []
    for b, p, e in zip(boxes, probs, emb):
        if p is not None and p >= min_prob:
            out.append({"box": [float(x) for x in b], "prob": float(p), "embedding": e})
    return out


def embed_largest_face(path) -> np.ndarray | None:
    """Training helper: embedding of the most confident face in an image file."""
    faces = detect_and_embed(Image.open(path))
    if not faces:
        return None
    return max(faces, key=lambda f: f["prob"])["embedding"]
