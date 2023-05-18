from typing import Tuple
import numpy as np
from pandas import DataFrame


def get_closed_form_solution(train_data: DataFrame) -> Tuple[float, float]:
    x_data = (train_data.index - train_data.index[0]).days
    y_data = train_data['Close'].values

    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    xy_mean = np.mean(x_data * y_data)
    x_squared_mean = np.mean(x_data ** 2)
    a = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
    b = y_mean - a * x_mean
    return a, b


def predict_price(test_data: DataFrame, train_data: DataFrame, a: float, b: float) -> DataFrame:
    x_test = (test_data.index - train_data.index[0]).days
    y_pred = a * x_test + b
    return y_pred


class LinearRegression:

    def __init__(self, train_data: DataFrame):
        self.train_data = train_data
        self.a, self.b = get_closed_form_solution(self.train_data)

    def predict(self, test_data: DataFrame):
        x_test = (test_data.index - self.train_data.index[0]).days
        y_pred = self.a * x_test + self.b
        return y_pred
