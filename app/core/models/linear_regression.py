from typing import Tuple
import numpy as np
from pandas import DataFrame


def get_closed_form_solution(train_data: DataFrame) -> Tuple[float, float]:
    x_data = (train_data.index - train_data.index[0]).days  # Adjust dates to start from zero
    y_data = train_data['Close'].values

    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    xy_mean = np.mean(x_data * y_data)
    x_squared_mean = np.mean(x_data ** 2)
    a = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
    b = y_mean - a * x_mean
    return a, b
