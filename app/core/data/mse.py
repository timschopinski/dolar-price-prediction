import numpy as np
from pandas import DataFrame


def calculate_mse(a: float, b: float, data: DataFrame) -> float:
    x_data = (data.index - data.index[0]).days  # Adjust dates to start from zero
    y_data = data['Close'].values

    y_pred = a * x_data + b
    mse = np.mean((y_data - y_pred) ** 2)
    return float(mse)
