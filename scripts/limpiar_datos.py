"""Validaciones y transformaciones sin modificar el DataFrame original."""

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


CATEGORIES = (
    "Media & Entertainment", "Creative Arts", "Design",
    "Literature & Journalism", "Food",
)
MEDIA_BINARY = ("Video_Header", "Videos_Bool", "Images_Bool")
LOG_TRANSFORMS = {
    "Ln1p_Funding": "Funding", "Ln1p_Target": "Target",
    "Ln1p_Prev_Success": "Prev_Success", "Ln1p_Backed": "Backed",
}
OUTLIER_VARIABLES = (
    "Funding", "Target", "Duration_Calendar", "Description",
    "Rewards", "Prev_Success", "Backed", "Links",
)
CORE_COLUMNS = {
    "ID", "Start_Date", "End_Date", "Duration", "Category", "Category_Kickstarter",
    "Success", "Funding", "Funding_IHS", "Target", "Target_IHS", "Staff_Pick",
    "Backers", *MEDIA_BINARY, "Description", "Rewards", "Prev_Success", "Backed",
    "Updates", "Links", "EU_Backers_Tot", "EU_Backers_Share", "Same_Ctry_Tot",
    "Same_Ctry_Share", "Creator_Location", "Creator_Coord", "Avg_Dist_EU",
    "Country_Pop", "lnCountry_Pop", "Adjacent_Ctry_Tot", "Adjacent_Ctry_Share",
    "Adjacent_Ctry_Backers_Sum", "Adjacent_Ctry_Count",
}
DUMMY_NAMES = ["Category_" + c.replace(" & ", "_").replace(" ", "_") for c in CATEGORIES[1:]]
PREDICTORS = [
    "Ln1p_Target", "Duration_Calendar", "Description", "Rewards",
    "Ln1p_Prev_Success", "Ln1p_Backed", "Links", *MEDIA_BINARY, *DUMMY_NAMES,
]


@dataclass
class PreparationResult:
    raw: pd.DataFrame
    prepared: pd.DataFrame
    model_data: pd.DataFrame
    audit: dict
    outliers: pd.DataFrame
    country_columns: list[str]
    origin_columns: list[str]


def require(condition: bool, message: str) -> None:
    """Las validaciones no desaparecen al ejecutar Python con -O."""
    if not condition:
        raise ValueError(message)


def geographical_columns(raw: pd.DataFrame) -> tuple[list[str], list[str]]:
    origin = [c for c in raw if c.endswith("_Bool") and c not in MEDIA_BINARY]
    countries = [c for c in raw if c not in CORE_COLUMNS and c not in origin]
    require(len(origin) == 27 and len(countries) == 138,
            "El esquema geográfico difiere de la versión 1: se esperan 27 indicadores y 138 conteos.")
    return countries, origin


def prepare(raw: pd.DataFrame) -> PreparationResult:
    """Auditar y crear 23 columnas; no imputar, recortar ni eliminar registros."""
    require(not raw.empty, "La base está vacía.")
    require(raw.columns.is_unique, "Hay encabezados duplicados.")
    require(CORE_COLUMNS.issubset(raw.columns), "Faltan variables del esquema original EU.")
    require(len(raw.columns) == 200, "La versión 1 EU debe contener 200 variables.")
    require(not raw.isna().any().any(), "Hay faltantes originales: deben revisarse antes de continuar.")
    require(not raw.duplicated().any(), "Hay filas duplicadas; se requiere revisión por ID.")
    require(raw.ID.is_unique, "Hay identificadores repetidos; no se eliminan automáticamente.")
    require(set(raw.Category.unique()).issubset(CATEGORIES), "Category contiene valores desconocidos.")
    countries, origin = geographical_columns(raw)
    numeric = raw.select_dtypes("number")
    require(np.isfinite(numeric.to_numpy()).all(), "Hay valores numéricos no finitos.")
    require(numeric.ge(0).all().all(), "Hay valores numéricos negativos.")
    binary = ["Success", "Staff_Pick", *MEDIA_BINARY, *origin]
    require(raw[binary].isin([0, 1]).all().all(), "Una variable binaria tiene valores distintos de 0/1.")
    require(raw[origin].sum(axis=1).eq(1).all(), "Cada campaña debe tener un solo país de origen.")
    counts = [*countries, "Backers", "EU_Backers_Tot", "Same_Ctry_Tot", "Adjacent_Ctry_Tot",
              "Duration", "Description", "Rewards", "Prev_Success", "Backed", "Updates", "Links"]
    require(raw[counts].mod(1).eq(0).all().all(), "Se encontraron conteos o duraciones no enteros.")
    require(raw.Backers.gt(0).all(), "Hay campañas sin patrocinadores totales: revisar EU_Backers_Share.")
    require(raw.EU_Backers_Tot.le(raw.Backers).all(), "Un total UE supera Backers.")
    require(raw[["Same_Ctry_Tot", "Adjacent_Ctry_Tot"]].le(raw.EU_Backers_Tot, axis=0).all().all(),
            "Un numerador geográfico supera el total UE.")

    d = raw.copy(deep=True)
    for column in ["Start_Date", "End_Date"]:
        d[column] = pd.to_datetime(raw[column], format="%Y-%m-%d", errors="raise")
    require(d.End_Date.ge(d.Start_Date).all(), "Hay campañas que cierran antes de comenzar.")
    d["Duration_Calendar"] = (d.End_Date - d.Start_Date).dt.days
    d["Flag_Duration_Discrepancy"] = d.Duration.ne(d.Duration_Calendar).astype(int)
    d["Flag_Zero_Funding"] = d.Funding.eq(0).astype(int)
    d["Flag_No_EU_Backers"] = d.EU_Backers_Tot.eq(0).astype(int)
    # La validación one-hot precede a idxmax para evitar asignar países a filas inválidas.
    d["Creator_Country_Derived"] = raw[origin].idxmax(axis=1).str.removesuffix("_Bool")
    denominator = d.EU_Backers_Tot.replace(0, np.nan)
    d["Same_Ctry_Share_Clean"] = d.Same_Ctry_Tot / denominator
    d["Adjacent_Ctry_Share_Clean"] = d.Adjacent_Ctry_Tot / denominator
    for new, original in LOG_TRANSFORMS.items():
        d[new] = np.log1p(raw[original])

    outliers = []
    for column in OUTLIER_VARIABLES:
        q1, q3 = d[column].quantile([0.25, 0.75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        flag = d[column].lt(lower) | d[column].gt(upper)
        d["Flag_IQR_" + column] = flag.astype(int)
        outliers.append({"Variable": column, "Q1": q1, "Q3": q3,
                         "Limite_inferior": lower, "Limite_superior": upper,
                         "Casos_senalados": int(flag.sum()), "Eliminados": 0})
    for category, name in zip(CATEGORIES[1:], DUMMY_NAMES):
        d[name] = raw.Category.eq(category).astype(int)
    model_data = d[["ID", "Ln1p_Funding", "Success", *PREDICTORS]].copy()
    positive_eu = raw.EU_Backers_Tot.gt(0)
    checks = {
        "success_coincide_con_meta": bool(raw.Success.eq(raw.Funding.ge(raw.Target).astype(int)).all()),
        "funding_ihs_coincide_asinh": bool(np.allclose(raw.Funding_IHS, np.arcsinh(raw.Funding))),
        "target_ihs_coincide_asinh": bool(np.allclose(raw.Target_IHS, np.arcsinh(raw.Target))),
        "ln_poblacion_coincide": bool(np.allclose(raw.lnCountry_Pop, np.log(raw.Country_Pop))),
        "eu_share_coincide": bool(np.allclose(raw.EU_Backers_Share, raw.EU_Backers_Tot / raw.Backers)),
        "same_share_coincide_con_denominador_positivo": bool(np.allclose(
            raw.loc[positive_eu, "Same_Ctry_Share"], d.loc[positive_eu, "Same_Ctry_Share_Clean"])),
        "adjacent_share_coincide_con_denominador_positivo": bool(np.allclose(
            raw.loc[positive_eu, "Adjacent_Ctry_Share"], d.loc[positive_eu, "Adjacent_Ctry_Share_Clean"])),
        "sin_faltantes_matriz_modelo": bool(model_data.notna().all().all()),
        "matriz_modelo_finita": bool(np.isfinite(model_data.to_numpy()).all()),
        "matriz_con_intercepto_rango_completo": bool(np.linalg.matrix_rank(
            np.column_stack([np.ones(len(d)), model_data[PREDICTORS].to_numpy(float)])) == len(PREDICTORS) + 1),
    }
    for name, valid in checks.items():
        require(valid, f"Falló el control: {name}")
    country_sum = raw[countries].sum(axis=1)
    difference = raw.Duration - d.Duration_Calendar
    audit = {
        "filas_originales": len(raw), "columnas_originales": len(raw.columns),
        "faltantes_originales": int(raw.isna().sum().sum()), "duplicados_completos": int(raw.duplicated().sum()),
        "ids_duplicados": int(raw.ID.duplicated().sum()), "no_finitos_originales": 0,
        "columnas_constantes": list(raw.columns[raw.nunique(dropna=False).le(1)]),
        "columnas_patrocinadores_por_pais": len(countries), "columnas_dummy_pais_origen": len(origin),
        "controles": checks, "filas_finales": len(d), "columnas_finales": len(d.columns),
        "columnas_derivadas": len(d.columns) - len(raw.columns),
        "columnas_matriz_modelo": len(model_data.columns), "numero_predictores": len(PREDICTORS),
        "predictores": PREDICTORS,
        "respuestas": {"lineal": "Ln1p_Funding = ln(1 + Funding)", "logistica": "Success (0/1)"},
        "fechas_observadas": {c: [d[c].min().strftime("%Y-%m-%d"), d[c].max().strftime("%Y-%m-%d")]
                              for c in ["Start_Date", "End_Date"]},
        "success_conteos": {str(k): int(v) for k, v in raw.Success.value_counts().sort_index().items()},
        "duration_discrepancias": int(d.Flag_Duration_Discrepancy.sum()),
        "duration_original_menos_calculada": {str(k): int(v) for k, v in difference.value_counts().items()},
        "sin_patrocinadores_ue": int(d.Flag_No_EU_Backers.sum()),
        "recaudacion_cero": int(d.Flag_Zero_Funding.sum()),
        "recaudacion_cero_con_patrocinadores": int((raw.Funding.eq(0) & raw.Backers.gt(0)).sum()),
        "faltantes_nuevos_documentados": {c: int(d[c].isna().sum())
                                           for c in ["Same_Ctry_Share_Clean", "Adjacent_Ctry_Share_Clean"]},
        "casos_suma_paises_menor_backers": int(country_sum.lt(raw.Backers).sum()),
        "casos_suma_paises_mayor_backers": int(country_sum.gt(raw.Backers).sum()),
        "paises_con_observaciones": int(d.Creator_Country_Derived.nunique()),
        "filas_eliminadas": 0, "imputaciones": 0, "winsorizaciones": 0,
        "estandarizacion": False, "balanceo_clases": False, "modelos_ajustados": False,
    }
    return PreparationResult(raw.copy(deep=True), d, model_data, audit,
                             pd.DataFrame(outliers), countries, origin)


def save_csv(frame, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, encoding="utf-8-sig", date_format="%Y-%m-%d", na_rep="")


def run_pipeline(source: Path, processed_dir: Path, output_dir: Path, metadata_dir: Path):
    original_bytes = source.read_bytes()
    raw = pd.read_csv(source, encoding="utf-8-sig")
    schema = pd.read_csv(metadata_dir / "diccionario_original.csv")
    require(list(raw.columns) == schema.Variable.to_list(), "Los encabezados no coinciden con el diccionario de la versión 1.")
    result = prepare(raw)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir
    tables.mkdir(parents=True, exist_ok=True)
    d = result.prepared
    save_csv(d, processed_dir / "eu_limpio.csv")
    save_csv(result.model_data, processed_dir / "eu_modelos.csv")
    save_csv(result.outliers, tables / "auditoria_atipicos.csv")
    save_csv(d.loc[d.Flag_Duration_Discrepancy.eq(1), ["ID", "Start_Date", "End_Date", "Duration", "Duration_Calendar"]], tables / "discrepancias_duracion.csv")
    save_csv(d.loc[d.Flag_Zero_Funding.eq(1), ["ID", "Funding", "Target", "Success", "Backers"]], tables / "revision_recaudacion_cero.csv")
    save_csv(d.loc[d.Flag_No_EU_Backers.eq(1), ["ID", "EU_Backers_Tot", "Same_Ctry_Share", "Same_Ctry_Share_Clean", "Adjacent_Ctry_Share", "Adjacent_Ctry_Share_Clean", "Avg_Dist_EU"]], tables / "revision_sin_patrocinadores_ue.csv")
    dictionaries = []
    for name, frame in [("original", raw), ("derivadas", d.drop(columns=raw.columns))]:
        dictionary = pd.read_csv(metadata_dir / f"diccionario_{name}.csv")
        require(set(dictionary.Variable) == set(frame.columns), f"Diccionario {name} incompleto.")
        for statistic, values in {"Faltantes": frame.isna().sum(), "Valores_distintos": frame.nunique(),
                                  "Constante": frame.nunique(dropna=False).le(1)}.items():
            dictionary[statistic] = dictionary.Variable.map(values)
        dictionaries.append(dictionary)
    save_csv(pd.concat(dictionaries, ignore_index=True), processed_dir / "diccionario_eu_limpio.csv")
    require(source.read_bytes() == original_bytes, "El archivo original cambió durante la ejecución.")
    result.audit.update({"archivo": source.name, "sha256": hashlib.sha256(original_bytes).hexdigest(),
                         "original_sin_modificaciones": True,
                         "entorno": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}})
    (output_dir / "auditoria.json").write_text(json.dumps(result.audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Auditar y preparar las campañas de la UE; no genera documentos.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Raíz del repositorio (por defecto, raíz del proyecto)")
    args = parser.parse_args()
    root = args.root.resolve()
    result = run_pipeline(root / "data/cruda/crowdfunding_eu_data.csv", root / "data/procesada",
                          root / "resultados/limpieza", root / "data/diccionarios")
    audit = result.audit
    print(f"Original: {audit['filas_originales']} x {audit['columnas_originales']}")
    print(f"Preparado: {audit['filas_finales']} x {audit['columnas_finales']}; matriz inicial: {result.model_data.shape}")
    print(f"Duraciones discrepantes: {audit['duration_discrepancias']}; sin patrocinadores UE: {audit['sin_patrocinadores_ue']}; recaudaciones cero: {audit['recaudacion_cero']}")
    print(f"Salidas: {root / 'resultados/limpieza'}")


if __name__ == "__main__":
    main()
