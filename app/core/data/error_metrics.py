import numpy as np


def calculate_mse(y_pred: np.ndarray, y_actual: np.ndarray) -> float:
    mse = np.mean((y_actual - y_pred) ** 2)
    return float(mse)


def calculate_mae(y_pred: np.ndarray, y_actual: np.ndarray) -> float:
    mae = np.mean(np.abs(y_actual - y_pred))
    return float(mae)


def calculate_rmse(y_pred: np.ndarray, y_actual: np.ndarray) -> float:
    mse = np.mean((y_actual - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return float(rmse)
