import numpy as np
from pandas import DataFrame


def calculate_mse(y_pred: DataFrame, y_actual: DataFrame) -> float:
    mse = np.mean((y_actual - y_pred) ** 2)
    return float(mse)
