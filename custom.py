"""Classes próprias usadas no pipeline.

IMPORTANTE: o pickle/joblib guarda o caminho da classe (ex.: custom.PrinceGPA).
Por isso ela precisa viver aqui e não dentro do notebook (onde vira __main__.PrinceGPA
e não pode ser carregada pelo Streamlit).
"""
import numpy as np
import prince
from sklearn.base import BaseEstimator, TransformerMixin


class PrinceGPA(BaseEstimator, TransformerMixin):
    """Alinhamento Procrustes generalizado (translação, escala e rotação)."""

    def fit(self, X, y=None):
        self.gpa_ = prince.GPA().fit(np.asarray(X).reshape(len(X), -1, 2))
        return self

    def transform(self, X):
        S = self.gpa_.transform(np.asarray(X).reshape(len(X), -1, 2))
        return np.asarray(S).reshape(len(S), -1)