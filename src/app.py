"""Proyecto 4Geeks: sistema de prediccion de ventas con ARIMA."""

from itertools import product
from pathlib import Path
import pickle
import warnings

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=2546&path=sales.csv"
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "sales.csv"
SERIES_PATH = ROOT_DIR / "data" / "processed" / "sales_time_series.csv"
GRID_RESULTS_PATH = ROOT_DIR / "data" / "processed" / "arima_grid_results.csv"
TEST_FORECAST_PATH = ROOT_DIR / "data" / "processed" / "arima_test_forecast.csv"
FUTURE_FORECAST_PATH = ROOT_DIR / "data" / "processed" / "arima_future_forecast_60_days.csv"
METRICS_PATH = ROOT_DIR / "data" / "processed" / "arima_metrics.csv"
MODEL_PATH = ROOT_DIR / "models" / "sales_arima_model.pkl"


def load_data() -> pd.DataFrame:
    """Load the local sales CSV, falling back to the public URL."""
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)

    return pd.read_csv(DATA_URL)


def prepare_series(df: pd.DataFrame) -> pd.Series:
    """Create a daily time series indexed by date."""
    series_df = df.copy()
    series_df["date"] = pd.to_datetime(series_df["date"]).dt.normalize()
    series_df = series_df.sort_values("date").drop_duplicates(subset="date")
    series = series_df.set_index("date")["sales"].asfreq("D")
    return series.interpolate(method="time")


def search_arima_order(train: pd.Series) -> tuple[tuple[int, int, int], pd.DataFrame]:
    """Search a compact ARIMA grid using AIC."""
    results = []

    for p, d, q in product(range(0, 4), range(1, 3), range(0, 6)):
        try:
            model = ARIMA(
                train,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit()
            results.append({"p": p, "d": d, "q": q, "aic": fitted.aic, "bic": fitted.bic})
        except Exception:
            continue

    results_df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)
    best_order = tuple(results_df.loc[0, ["p", "d", "q"]].astype(int))
    return best_order, results_df


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    series = prepare_series(df)
    series.to_frame("sales").to_csv(SERIES_PATH)

    train_size = int(len(series) * 0.8)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    best_order, grid_results = search_arima_order(train)
    grid_results.to_csv(GRID_RESULTS_PATH, index=False)

    model = ARIMA(
        train,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=len(test))

    metrics = {
        "best_order": str(best_order),
        "tensor": "daily",
        "adf_statistic_original": adfuller(series.dropna())[0],
        "adf_pvalue_original": adfuller(series.dropna())[1],
        "adf_statistic_diff1": adfuller(series.diff().dropna())[0],
        "adf_pvalue_diff1": adfuller(series.diff().dropna())[1],
        "mae": mean_absolute_error(test, forecast),
        "rmse": mean_squared_error(test, forecast) ** 0.5,
        "mape": mean_absolute_percentage_error(test, forecast),
    }

    forecast_df = pd.DataFrame(
        {
            "date": test.index,
            "actual_sales": test.values,
            "predicted_sales": forecast.values,
            "absolute_error": abs(test.values - forecast.values),
        }
    )
    forecast_df.to_csv(TEST_FORECAST_PATH, index=False)

    final_model = ARIMA(
        series,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    future_forecast = final_model.forecast(steps=60)
    future_df = pd.DataFrame({"date": future_forecast.index, "forecast_sales": future_forecast.values})
    future_df.to_csv(FUTURE_FORECAST_PATH, index=False)

    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(final_model, model_file)

    print("Serie temporal guardada en:", SERIES_PATH)
    print("Resultados de busqueda ARIMA guardados en:", GRID_RESULTS_PATH)
    print("Predicciones de test guardadas en:", TEST_FORECAST_PATH)
    print("Prediccion futura guardada en:", FUTURE_FORECAST_PATH)
    print("Metricas guardadas en:", METRICS_PATH)
    print("Mejor orden ARIMA:", best_order)
    print(pd.DataFrame([metrics]).round(4).to_string(index=False))
    print("Modelo ARIMA guardado en:", MODEL_PATH)


if __name__ == "__main__":
    main()
