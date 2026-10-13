"""Extração de landmarks faciais com MediaPipe Face Landmarker."""
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw

MODEL_PATH = Path(__file__).parent / "assets" / "face_landmarker.task"


class LandmarkExtractor:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Modelo MediaPipe não encontrado: {MODEL_PATH}"
            )
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract(self, img_rgb: np.ndarray) -> np.ndarray | None:
        """Retorna landmarks (478, 2) em pixels, ou None se não houver rosto."""
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.detector.detect(mp_img)
        if not result.face_landmarks:
            return None
        h, w = img_rgb.shape[:2]
        landmarks = result.face_landmarks[0]
        return np.array(
            [[point.x * w, point.y * h] for point in landmarks],
            dtype=float,
        )


def draw_landmarks(
    img_rgb: np.ndarray, pts_px: np.ndarray, keep: list[int] | None = None
) -> np.ndarray:
    """Desenha os pontos. Os usados pelo modelo ficam em vermelho."""
    image = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(image)
    used = set(keep) if keep is not None else set(range(len(pts_px)))
    radius = max(1, min(image.size) // 400)
    for index, (x, y) in enumerate(pts_px):
        color = (0, 255, 0) if index in used else (160, 160, 160)
        draw.ellipse(
            (int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius),
            fill=color,
        )
    return np.asarray(image)
