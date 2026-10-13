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
    """Desenha os usados pelo modelo em verde e os não usados em vermelho."""
    image = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(image)
    used = set(keep) if keep is not None else set(range(len(pts_px)))
    radius = max(1, min(image.size) // 400)
    for index, (x, y) in enumerate(pts_px):
        color = (0, 255, 0) if index in used else (255, 0, 0)
        draw.ellipse(
            (int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius),
            fill=color,
        )
    return np.asarray(image)

def landmarks_usados(modelo, n_landmarks=478):
    """Retorna os índices (0-based) dos landmarks que o modelo realmente usa."""
    sel = getattr(modelo, "named_steps", {}).get("select")

    if sel is None:                       # sem seleção: usa todos
        return np.arange(n_landmarks)

    if hasattr(sel, "get_support"):       # GenericUnivariateSelect (SVC)
        cols = np.where(sel.get_support())[0]
    else:                                 # ColumnTransformer (modelo_tea)
        cols = np.asarray(sel.transformers_[0][2])

    return np.unique(cols // 2)           # cada landmark ocupa 2 colunas (x, y)