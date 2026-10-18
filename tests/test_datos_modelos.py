import pandas as pd
import pytest

from modelos.datos import cargar_xy


def test_modelos_comparten_campanas_y_explicativas_sin_respuestas():
    x_reg, y_reg, ids_reg = cargar_xy("regresion")
    x_clf, y_clf, ids_clf = cargar_xy("clasificacion")
    pd.testing.assert_frame_equal(x_reg, x_clf)
    pd.testing.assert_series_equal(ids_reg, ids_clf)
    assert x_reg.shape == (2272, 14)
    assert y_reg.name == "Ln1p_Funding"
    assert y_clf.name == "Success"
    assert set(y_clf.unique()) == {0, 1}
    assert not {"ID", "Success", "Ln1p_Funding", "Funding", "Funding_IHS"}.intersection(x_reg.columns)


def test_tarea_desconocida_no_asigna_una_respuesta_por_defecto():
    with pytest.raises(ValueError, match="tarea debe"):
        cargar_xy("random_forest")
