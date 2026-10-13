"""Extração de landmarks faciais (MediaPipe Face Mesh, 478 pontos)."""
import cv2
import numpy as np
import mediapipe as mp


class LandmarkExtractor:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # necessário para os 478 pontos (íris)
        )

    def extract(self, img_rgb: np.ndarray):
        """Retorna array (478, 2) em PIXELS, ou None se não houver rosto."""
        res = self.mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        h, w = img_rgb.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        return np.array([(p.x * w, p.y * h) for p in lm], dtype=float)


def draw_landmarks(img_rgb: np.ndarray, pts_px: np.ndarray, keep=None) -> np.ndarray:
    """Desenha os pontos na imagem. Pontos usados pelo modelo ficam em vermelho."""
    out = img_rgb.copy()
    usados = set(keep) if keep is not None else set(range(len(pts_px)))
    r = max(1, int(min(out.shape[:2]) / 400))
    for i, (x, y) in enumerate(pts_px):
        cor = (255, 0, 0) if i in usados else (160, 160, 160)
        cv2.circle(out, (int(x), int(y)), r, cor, -1)
    return out