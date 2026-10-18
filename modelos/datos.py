"""Carga de datos compartida por regresión lineal, logística y random forest."""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from scripts.limpiar_datos import PREDICTORS


BASE_PROCESADA = Path(__file__).resolve().parents[1] / "data/procesada/eu_limpio.csv"
RESPUESTAS = {"regresion": "Ln1p_Funding", "clasificacion": "Success"}


def cargar_xy(tarea: Literal["regresion", "clasificacion"], ruta: Path | str = BASE_PROCESADA):
    """Devolver X, y e ID separados; la partición de entrenamiento se hace después."""
    if tarea not in RESPUESTAS:
        raise ValueError("tarea debe ser 'regresion' o 'clasificacion'.")
    data = pd.read_csv(ruta, encoding="utf-8-sig")
    required = ["ID", RESPUESTAS[tarea], *PREDICTORS]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"La base procesada no contiene las columnas requeridas: {sorted(missing)}")
    if data.empty or not data.ID.is_unique:
        raise ValueError("La base debe tener campañas e identificadores únicos.")
    if not np.isfinite(data[required].to_numpy()).all():
        raise ValueError("Hay faltantes o valores no finitos en las variables del modelo.")
    if tarea == "clasificacion" and not data.Success.isin([0, 1]).all():
        raise ValueError("Success debe estar codificada en 0/1.")
    return data[PREDICTORS].copy(), data[RESPUESTAS[tarea]].copy(), data.ID.copy()
