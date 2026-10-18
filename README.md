# Crowdfunding UE

Limpieza y transformación de 2.272 campañas de Kickstarter de la Unión Europea. El equipo desarrollará el EDA y los modelos a partir de la base procesada.

## Qué archivo enviar ahora

**`data/procesada/eu_limpio.csv` es la base procesada principal: 2.272 filas y 223 columnas.** Conserva las 200 variables originales y añade 23 variables de preparación. Enviar este archivo junto con `data/procesada/diccionario_eu_limpio.csv` permite compartir la base completa y sus definiciones.

`data/procesada/eu_modelos.csv` es una versión reducida: 2.272 filas y 17 columnas. Contiene ID, las dos respuestas y 14 explicativas iniciales. Es útil para empezar los modelos y comparar resultados, pero la base completa sigue siendo `eu_limpio.csv`.

Los CSV usan separador coma, punto decimal y UTF-8 con BOM. Los vacíos de las dos proporciones `*_Share_Clean` significan que la razón no está definida porque el denominador es cero; no significan cero.

## Carpetas

```text
data/
  cruda/crowdfunding_eu_data.csv       Original: 2.272 x 200; no editar
  procesada/eu_limpio.csv             Base principal para compartir: 2.272 x 223
  procesada/eu_modelos.csv            Selección inicial para modelos: 2.272 x 17
  procesada/diccionario_eu_limpio.csv  Definición de las 223 variables
  diccionarios/                      Definiciones usadas por el script
  SOURCE.md                          Fuente y licencia del dataset
scripts/
  limpiar_datos.py                   Todo el tratamiento de datos y su exportación
modelos/
  datos.py                           Carga común de X, y e ID para todos los modelos
resultados/
  limpieza/                          Auditoría y casos que requieren revisión
tests/                               Pruebas automáticas
```

`__init__.py` identifica los módulos Python. `pyproject.toml` y `uv.lock` definen el entorno; `.github/` verifica el proyecto al subir cambios. Las bases cruda y procesada se incluyen en GitHub. `.venv`, las cachés y `.local` están excluidos; el archivo mundial se conserva únicamente en `.local/datos_mundiales/` y no interviene en el análisis EU.

## Cómo ejecutar

Desde la raíz del repositorio, con Python 3.11 o posterior y [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra dev
uv run python scripts/limpiar_datos.py
uv run pytest -q
```

El script reconstruye los tres archivos de `data/procesada/` y la auditoría. Acepta `--root RUTA` y también puede ejecutarse desde otra carpeta. Sin uv, crear un entorno virtual e instalar con `python -m pip install -e ".[dev]"`.

## Punto de partida para que el equipo haga el EDA

Leer la base completa, no la selección reducida de modelos. Desde la raíz del proyecto:

```python
import pandas as pd

datos = pd.read_csv("data/procesada/eu_limpio.csv", encoding="utf-8-sig")
```

Consultar `data/procesada/diccionario_eu_limpio.csv` para interpretar las columnas originales, las transformaciones y las banderas de revisión. Esta entrega no incluye scripts, notebooks, gráficos ni resultados del EDA.

## Cómo empezar los modelos

```python
from modelos.datos import cargar_xy

X, y, ids = cargar_xy("regresion")      # y = Ln1p_Funding
X, y, ids = cargar_xy("clasificacion")  # y = Success
```

`X` contiene las mismas 14 explicativas iniciales, sin ID ni respuestas. `ids` permite identificar las campañas y mantener las mismas particiones al comparar modelos.

| Modelo por desarrollar | Tarea | Respuesta inicial |
|---|---|---|
| Regresión lineal | Regresión | `Ln1p_Funding` |
| Regresión logística | Clasificación | `Success` |
| Random forest regresor | Regresión | `Ln1p_Funding`, para comparar con la lineal |
| Random forest clasificador | Clasificación | `Success`, para comparar con la logística |

Añadir cada implementación como un archivo en `modelos/` y guardar sus salidas en `resultados/modelos/`. Todavía no hay modelos entrenados ni partición entrenamiento/prueba. Para comparar modelos de una misma tarea, usar la misma respuesta, las mismas campañas de entrenamiento/prueba y las mismas métricas. Ajustar escaladores, selección de variables y remuestreo solo dentro de entrenamiento.

La selección inicial puede ampliarse con justificación. No usar automáticamente las 223 columnas como X: incluyen respuestas, versiones redundantes, indicadores de auditoría y medidas al cierre. La disponibilidad de todos los atributos al lanzamiento no está documentada; el alcance actual es retrospectivo.

## Limpieza aplicada

- El original se conserva y se comprueba mediante SHA-256. No se eliminaron campañas ni se imputaron montos.
- Se detectaron 191 diferencias de duración: se añadió `Duration_Calendar` y se preservó `Duration`.
- Se corrigió la interpretación de dos proporciones en las mismas 32 campañas sin patrocinadores UE. Al realizar el EDA, interpretar la distancia media únicamente cuando hay patrocinadores UE.
- Se conservaron y señalaron diez recaudaciones cero con patrocinadores y los extremos detectados con IQR.
- Se añadieron cuatro transformaciones `ln(1+x)` y cuatro dummies de categoría; referencia: `Media & Entertainment`.
- No se estandarizaron variables ni se balancearon clases. Los límites IQR son diagnósticos descriptivos y sus banderas no forman parte de X.

## Fuente

Fellenberg, F. (2025). *Reward-Based Crowdfunding Data* (Versión 1) [Conjunto de datos]. Mendeley Data. https://doi.org/10.17632/9r6v4hsc8x.1

Datos bajo CC BY 4.0. La procedencia, la atribución y las modificaciones se detallan en [data/SOURCE.md](data/SOURCE.md).
