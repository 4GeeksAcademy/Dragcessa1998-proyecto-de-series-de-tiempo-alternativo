
# Sistema de Predicción de Ventas con ARIMA

Este proyecto desarrolla un sistema de predicción de ventas utilizando técnicas de análisis de series temporales. El objetivo es estimar el ritmo futuro de ventas de una empresa para apoyar la toma de decisiones logísticas, especialmente la planificación del espacio necesario en un nuevo almacén.

La serie histórica muestra un crecimiento sostenido desde la creación de la empresa, por lo que se analiza su comportamiento temporal y se entrena un modelo ARIMA para realizar predicciones sobre ventas futuras.

## Objetivo del Proyecto

El objetivo principal es construir un modelo capaz de predecir ventas futuras a partir del comportamiento histórico de la serie.

Para ello se desarrolla el siguiente flujo:

1. Cargar el conjunto de datos.
2. Construir una estructura válida de serie temporal.
3. Analizar el comportamiento de la serie.
4. Responder preguntas clave:
   - ¿Cuál es el tensor de la serie temporal?
   - ¿Cuál es la tendencia?
   - ¿Es estacionaria?
   - ¿Existe variabilidad o presencia de ruido?
5. Entrenar un modelo ARIMA.
6. Predecir sobre el conjunto de test.
7. Medir el rendimiento del modelo.
8. Guardar el modelo entrenado.

## Dataset

El conjunto de datos utilizado es `sales.csv`.

Contiene dos columnas:

| Variable | Descripción |
| --- | --- |
| `date` | Fecha de la observación |
| `sales` | Ventas registradas |

La columna `date` se transforma a formato datetime y se utiliza como índice temporal. La columna `sales` representa la variable que se desea predecir.

## Construcción de la Serie Temporal

Después de cargar el dataset, se ordenan las fechas y se construye una serie temporal diaria.

La serie contiene 366 registros, desde septiembre de 2022 hasta septiembre de 2023.

| Característica | Resultado |
| --- | --- |
| Fecha inicial | 2022-09-03 |
| Fecha final | 2023-09-03 |
| Número de registros | 366 |
| Tensor temporal | Diario |
| Valores faltantes | 0 |

## Análisis de la Serie

### Tensor de la serie temporal

El tensor de la serie temporal es diario. Esto significa que la unidad mínima de tiempo para la cual se tienen datos es un día.

### Tendencia

La serie presenta una tendencia claramente creciente. Las ventas comienzan cerca de valores alrededor de 55 y alcanzan valores cercanos a 1000 al final del periodo analizado.

Esto indica que la empresa ha experimentado un crecimiento sostenido en sus ventas.

### Estacionariedad

La serie original no es estacionaria. Para comprobarlo se utilizó la prueba Augmented Dickey-Fuller.

Resultado sobre la serie original:

```text
ADF p-value: 0.9862
```

Como el p-value es mayor que 0.05, no se rechaza la hipótesis nula de no estacionariedad.

Después de aplicar una primera diferenciación, el resultado fue:

```text
ADF p-value primera diferencia: 0.0000
```

Esto indica que la serie diferenciada sí puede considerarse estacionaria.

### Variabilidad y ruido

La serie presenta variabilidad alrededor de la tendencia principal. Sin embargo, el patrón de crecimiento es claro y estable, por lo que el ruido no impide construir un modelo predictivo útil.

## Modelo ARIMA

Se entrenó un modelo ARIMA utilizando el conjunto de entrenamiento. Para encontrar la mejor parametrización, se probaron distintas combinaciones de los parámetros `p`, `d` y `q`, seleccionando el modelo con mejor AIC.

El mejor modelo encontrado fue:

```text
ARIMA(0, 2, 5)
```

Interpretación de los parámetros:

| Parámetro | Valor | Interpretación |
| --- | ---: | --- |
| `p` | 0 | No utiliza términos autorregresivos |
| `d` | 2 | Aplica dos diferenciaciones |
| `q` | 5 | Utiliza cinco términos de media móvil |

## Evaluación del Modelo

El conjunto de datos se dividió en entrenamiento y prueba. El modelo fue entrenado con el 80% inicial de la serie y evaluado sobre el 20% final.

Métricas obtenidas en el conjunto de test:

| Métrica | Valor |
| --- | ---: |
| MAE | 2.3390 |
| RMSE | 2.8889 |
| MAPE | 0.0026 |

El MAPE equivale aproximadamente a un error porcentual medio de 0.26%, lo que indica que el modelo sigue muy de cerca la evolución real de las ventas.

## Predicción Futura

Además de comparar las predicciones con el conjunto de test, se generó un pronóstico futuro de 60 días utilizando el modelo entrenado sobre toda la serie.

Ejemplo de los primeros valores pronosticados:

| Fecha | Predicción de ventas |
| --- | ---: |
| 2023-09-04 | 1002.21 |
| 2023-09-05 | 1004.83 |
| 2023-09-06 | 1007.43 |

Este pronóstico permite estimar la evolución esperada de las ventas y apoyar la planificación del nuevo almacén.

## Estructura del Proyecto

```text
Dragcessa1998-proyecto-de-series-de-tiempo-alternativo-main/
│
├── data/
│   ├── raw/
│   │   └── sales.csv
│   ├── processed/
│   │   ├── sales_time_series.csv
│   │   ├── arima_grid_results.csv
│   │   ├── arima_test_forecast.csv
│   │   ├── arima_future_forecast_60_days.csv
│   │   └── arima_metrics.csv
│   └── interim/
│
├── models/
│   └── sales_arima_model.pkl
│
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   └── utils.py
│
├── requirements.txt
├── README.md
└── README.es.md
```

## Archivos Principales

| Archivo | Descripción |
| --- | --- |
| `src/explore.ipynb` | Notebook con el análisis completo de la serie temporal |
| `src/app.py` | Script reproducible para entrenar el modelo y guardar resultados |
| `data/raw/sales.csv` | Dataset original |
| `data/processed/sales_time_series.csv` | Serie temporal procesada |
| `data/processed/arima_grid_results.csv` | Resultados de búsqueda de parámetros ARIMA |
| `data/processed/arima_test_forecast.csv` | Comparación entre ventas reales y ventas predichas |
| `data/processed/arima_future_forecast_60_days.csv` | Pronóstico futuro de 60 días |
| `data/processed/arima_metrics.csv` | Métricas finales del modelo |
| `models/sales_arima_model.pkl` | Modelo ARIMA entrenado |

## Cómo Ejecutar el Proyecto

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el script principal:

```bash
python src/app.py
```

También se puede abrir y ejecutar el notebook:

```text
src/explore.ipynb
```

El notebook contiene el desarrollo completo: carga de datos, construcción de la serie temporal, análisis visual, prueba de estacionariedad, búsqueda de parámetros ARIMA, predicción, evaluación y guardado del modelo.

## Tecnologías Utilizadas

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels
- Jupyter Notebook
- Pickle

## Conclusión

La serie de ventas presenta una tendencia creciente clara y no es estacionaria en su forma original. Después de aplicar diferenciación, se logra una estructura adecuada para entrenar un modelo ARIMA.

El modelo ARIMA(0, 2, 5) obtuvo un rendimiento sólido sobre el conjunto de test, con un MAPE aproximado de 0.26%. Esto lo convierte en una herramienta útil para estimar ventas futuras y apoyar decisiones de planificación logística, como dimensionar el espacio necesario para un nuevo almacén.
## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.
