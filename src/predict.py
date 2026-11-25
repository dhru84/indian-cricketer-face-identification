"""Inference: find every face in an image and name each one (or say Unknown)."""
from __future__ import annotations

import joblib
import numpy as np
from PIL import Image, ImageDraw

from .config import MODEL_PATH, UNKNOWN_LABEL
from .embeddings import detect_and_embed


class FaceIdentifier:
    def __init__(self, model_path=MODEL_PATH, threshold: float | None = None):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.classes = bundle["classes"]
        self.threshold = bundle["threshold"] if threshold is None else threshold

    def identify(self, img: Image.Image, top_k: int = 3) -> list[dict]:
        out = []
        for face in detect_and_embed(img):
            proba = self.model.predict_proba(face["embedding"][None])[0]
            order = np.argsort(proba)[::-1][:top_k]
            top = [(str(self.classes[i]), float(proba[i])) for i in order]
            name = top[0][0] if top[0][1] >= self.threshold else UNKNOWN_LABEL
            out.append({"box": face["box"], "name": name, "confidence": top[0][1], "top_k": top})
        return out

    @staticmethod
    def annotate(img: Image.Image, results: list[dict]) -> Image.Image:
        img = img.convert("RGB").copy()
        draw = ImageDraw.Draw(img)
        for r in results:
            x1, y1, x2, y2 = r["box"]
            color = "#22c55e" if r["name"] != UNKNOWN_LABEL else "#ef4444"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 250))
            draw.text((x1 + 4, max(0, y1 - 14)), f"{r['name']} {r['confidence']:.0%}", fill=color)
        return img
