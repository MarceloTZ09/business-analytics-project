from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.limpiar_datos import CATEGORIES, DUMMY_NAMES, PREDICTORS, prepare


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(ROOT / "data/cruda/crowdfunding_eu_data.csv")


@pytest.fixture(scope="module")
def result(raw):
    return prepare(raw)


def test_source_is_preserved_and_no_rows_are_removed(raw, result):
    pd.testing.assert_frame_equal(raw, result.raw)
    pd.testing.assert_frame_equal(raw.drop(columns=["Start_Date", "End_Date"]),
                                  result.prepared[raw.columns].drop(columns=["Start_Date", "End_Date"]))
    assert result.prepared.ID.to_list() == raw.ID.to_list()
    assert result.prepared.shape == (2272, 223)


def test_zero_denominators_remain_undefined(result):
    d = result.prepared
    mask = d.EU_Backers_Tot.eq(0)
    assert mask.sum() == 32
    assert d.loc[mask, ["Same_Ctry_Share_Clean", "Adjacent_Ctry_Share_Clean"]].isna().all().all()
    assert d.loc[~mask, ["Same_Ctry_Share_Clean", "Adjacent_Ctry_Share_Clean"]].notna().all().all()
    assert d.Same_Ctry_Share.notna().all()  # La columna publicada sigue intacta.


def test_duration_uses_dates_without_overwriting_reported_value(result):
    d = result.prepared
    pd.testing.assert_series_equal(d.Duration_Calendar, (d.End_Date - d.Start_Date).dt.days, check_names=False)
    assert (d.Duration - d.Duration_Calendar).value_counts().to_dict() == {0: 2081, -1: 186, -7: 4, -25: 1}


def test_zero_funding_is_kept_in_log_scale(result):
    zeros = result.prepared.Funding.eq(0)
    assert zeros.sum() == 10
    assert result.prepared.loc[zeros, "Ln1p_Funding"].eq(0).all()
    assert np.isfinite(result.model_data.to_numpy()).all()


def test_reference_category_and_no_outcome_leakage(result):
    d = result.prepared
    assert d.loc[d.Category.eq(CATEGORIES[0]), DUMMY_NAMES].eq(0).all().all()
    assert d.loc[d.Category.ne(CATEGORIES[0]), DUMMY_NAMES].sum(axis=1).eq(1).all()
    assert not {"ID", "Success", "Funding", "Ln1p_Funding", "Funding_IHS", "Backers", "Updates", "Staff_Pick"}.intersection(PREDICTORS)
    assert result.model_data.columns.to_list() == ["ID", "Ln1p_Funding", "Success", *PREDICTORS]


def test_invalid_country_and_unknown_categories_are_not_silently_recoded(raw):
    invalid = raw.copy()
    origins = [c for c in invalid if c.endswith("_Bool") and c not in ["Videos_Bool", "Images_Bool"]]
    invalid.loc[invalid.index[0], origins] = 0
    with pytest.raises(ValueError, match="país de origen"):
        prepare(invalid)
    invalid = raw.copy()
    invalid.loc[invalid.index[0], "Category"] = "Categoria no documentada"
    with pytest.raises(ValueError, match="desconocidos"):
        prepare(invalid)
