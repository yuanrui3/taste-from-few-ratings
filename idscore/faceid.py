"""Face likeness: YuNet detector (MIT) + SFace embedding (Apache-2.0), both
from the OpenCV model zoo, run through cv2.dnn on CPU.

likeness(render_face, real_face) = cosine similarity of SFace embeddings.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

YUNET_URL = "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"

# canonical 5-point template SFace was trained with (112x112), used for the
# landmark-free fallback crop
_TEMPLATE_BOX = 112


class FaceID:
    def __init__(self, model_dir: str, det_score: float = 0.5, second=None):
        self.second = second  # optional second embedder with an embed_crop(rgb) method
        self.det_path = os.path.join(model_dir, os.path.basename(YUNET_URL))
        self.rec_path = os.path.join(model_dir, os.path.basename(SFACE_URL))
        for p in (self.det_path, self.rec_path):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        self.det = cv2.FaceDetectorYN.create(self.det_path, "", (320, 320), det_score, 0.3, 50)
        self.rec = cv2.FaceRecognizerSF.create(self.rec_path, "")

    def detect(self, rgb: np.ndarray, face_box=None, upscale_to: int = 480):
        """Detect the most confident face.

        With `face_box` (x0, y0, x1, y1) the search runs on a window around
        that box, enlarged so the face spans ~130 px: landmarks on small
        faces are then far more precise than a whole-person detection.
        Returns the YuNet row (x, y, w, h, 5 landmarks, score) in the
        coordinates of `rgb`, or None.
        """
        H, W = rgb.shape[:2]
        ox = oy = 0
        img = rgb
        if face_box is not None:
            x0, y0, x1, y1 = face_box
            m = int(round(0.9 * max(x1 - x0, y1 - y0)))
            ox, oy = max(0, x0 - m), max(0, y0 - m)
            img = rgb[oy : min(H, y1 + m), ox : min(W, x1 + m)]
            f = 130.0 / max(1, max(x1 - x0, y1 - y0))
            f = float(np.clip(f, 1.0, 6.0))
        else:
            f = max(1.0, upscale_to / max(H, W))
        if img.size == 0:
            return None
        if f > 1:
            img = cv2.resize(img, (int(round(img.shape[1] * f)), int(round(img.shape[0] * f))), interpolation=cv2.INTER_CUBIC)
        bgr = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        self.det.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        row = faces[np.argmax(faces[:, 14])].copy()
        row[:14] /= f
        row[0] += ox
        row[1] += oy
        row[4:14:2] += ox
        row[5:14:2] += oy
        return row

    def _crop(self, rgb: np.ndarray, face_row=None, fallback_box=None):
        """112x112 BGR face crop. Landmark-aligned with a detection row; otherwise
        a square crop around `fallback_box`. Returns (crop, mode) or (None, "none")."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if face_row is not None:
            return self.rec.alignCrop(bgr, face_row.reshape(1, -1).astype(np.float32)), "aligned"
        if fallback_box is not None:
            x0, y0, x1, y1 = fallback_box
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            half = 0.62 * max(x1 - x0, y1 - y0)
            k = _TEMPLATE_BOX / (2 * half)
            M = np.array([[k, 0, _TEMPLATE_BOX / 2 - cx * k], [0, k, _TEMPLATE_BOX / 2 - cy * k]])
            return cv2.warpAffine(bgr, M, (_TEMPLATE_BOX, _TEMPLATE_BOX), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE), "box"
        return None, "none"

    def embed(self, rgb: np.ndarray, face_row=None, fallback_box=None):
        """SFace embedding (L2-normalised, 128-d) of the face crop.
        Returns (embedding, mode) with mode in {"aligned", "box"}; (None, "none") if no crop."""
        crop, mode = self._crop(rgb, face_row, fallback_box)
        if crop is None:
            return None, mode
        e = self.rec.feature(crop).reshape(-1).astype(np.float64)
        return e / (np.linalg.norm(e) + 1e-12), mode

    def embed2(self, rgb: np.ndarray, face_row=None, fallback_box=None):
        """Embedding of the same crop from the optional second model."""
        if self.second is None:
            return None
        crop, _ = self._crop(rgb, face_row, fallback_box)
        if crop is None:
            return None
        return self.second.embed_crop(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


def cosine(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.dot(a, b))
