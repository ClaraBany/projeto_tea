"""Leitura de landmarks colados manualmente na aplicação."""
import re

import numpy as np


def parse_landmarks(text: str) -> np.ndarray:
    """Lê 478 pares x,y ou 956 valores e retorna um array (478, 2)."""
    content = text.strip()
    if len(content) >= 2 and content[0] == content[-1] == '"':
        content = content[1:-1].replace('""', '"')

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    values = []
    for line in lines:
        tokens = [token for token in re.split(r"[,;\s]+", line) if token]
        try:
            values.extend(float(token) for token in tokens)
        except ValueError as e:
            raise ValueError(
                "Use apenas números, separados por vírgula, ponto e vírgula "
                "ou espaços."
            ) from e

    expected = 478 * 2
    if len(values) != expected:
        raise ValueError(
            f"São necessários 478 pares (956 números); foram encontrados "
            f"{len(values)} números."
        )
    points = np.asarray(values, dtype=float).reshape(478, 2)
    if not np.isfinite(points).all():
        raise ValueError("Os landmarks devem conter apenas números finitos.")
    return points
