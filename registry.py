"""Registro de modelos: qualquer .joblib dentro de models/ vira uma opção no app.

Configuração opcional: um .json com o MESMO nome do modelo, por exemplo:
    models/svc_v1.joblib
    models/svc_v1.json
"""
import json
import __main__
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import custom

# Modelos antigos exportados do notebook podem referenciar __main__.PrinceGPA.
# Redireciona essa referência para a implementação importável antes do joblib.load.
__main__.PrinceGPA = custom.PrinceGPA

MODELS_DIR = Path(__file__).parent / "models"

DEFAULT_CONFIG = {
    "descricao": "",
    "coords": "pixel",        # "pixel" ou "normalized" (x/largura, y/altura)
    "keep_indices": None,     # lista de índices do MediaPipe, ou None = todos os 478
    "labels": None,           # opcional: lista de nomes das classes
}


@dataclass
class ModelBundle:
    name: str
    pipeline: object
    config: dict

    def prepare(self, pts_px: np.ndarray, w: int, h: int) -> np.ndarray:
        """Converte (478, 2) em pixels no vetor (1, n_features) que o modelo espera."""
        pts = pts_px.copy()
        if self.config["coords"] == "normalized":
            pts[:, 0] /= w
            pts[:, 1] /= h
        idx = self.config["keep_indices"]
        if idx is not None:
            pts = pts[idx]
        return pts.reshape(1, -1)  # x0, y0, x1, y1, ...

    def predict(self, X: np.ndarray):
        """Retorna (classe, {classe: prob} ou None, score ou None).

        score = distância com sinal à fronteira de decisão (usado quando o modelo
        não tem predict_proba, como um SVC treinado sem probability=True).
        """
        pred = self.pipeline.predict(X)[0]
        labels = self.config["labels"]
        nome = labels[int(pred)] if labels is not None else pred

        probs, score = None, None
        if hasattr(self.pipeline, "predict_proba"):
            try:
                p = self.pipeline.predict_proba(X)[0]
                classes = labels if labels is not None else self.pipeline.classes_
                probs = {str(c): float(v) for c, v in zip(classes, p)}
            except Exception:
                probs = None
        if probs is None and hasattr(self.pipeline, "decision_function"):
            try:
                d = np.ravel(self.pipeline.decision_function(X))
                score = float(d[0]) if d.size == 1 else None
            except Exception:
                score = None
        return nome, probs, score


def list_models() -> list[str]:
    MODELS_DIR.mkdir(exist_ok=True)
    return sorted(p.stem for p in MODELS_DIR.glob("*.joblib"))


def load_model(name: str) -> ModelBundle:
    pipeline = joblib.load(MODELS_DIR / f"{name}.joblib")
    config = dict(DEFAULT_CONFIG)
    cfg_path = MODELS_DIR / f"{name}.json"
    if cfg_path.exists():
        config.update(json.loads(cfg_path.read_text(encoding="utf-8")))
    return ModelBundle(name=name, pipeline=pipeline, config=config)