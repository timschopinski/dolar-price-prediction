from typing import List, Tuple
import numpy as np
from numpy import ndarray
from pandas import DataFrame


def get_segmented_regression(train_data: DataFrame) -> List[Tuple[float, float]]:
    x_data = (train_data.index - train_data.index[0]).days
    y_data = train_data['Close'].values

    # Podział danych na segmenty
    segments = []
    current_segment = [0]  # Indeksy początków segmentów
    for i in range(1, len(y_data)):
        if np.sign(y_data[i] - y_data[i-1]) != np.sign(y_data[i-1] - y_data[i-2]):
            current_segment.append(i)
    current_segment.append(len(y_data))

    # Dopasowanie regresji liniowej do każdego segmentu
    regression_params = []
    for i in range(len(current_segment) - 1):
        start = current_segment[i]
        end = current_segment[i+1]
        segment_x = x_data[start:end]
        segment_y = y_data[start:end]
        segment_a, segment_b = get_closed_form_solution(segment_x, segment_y)
        regression_params.append((segment_a, segment_b))

    return regression_params


def get_closed_form_solution(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[float, float]:
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    xy_mean = np.mean(x_data * y_data)
    x_squared_mean = np.mean(x_data ** 2)
    a = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
    b = y_mean - a * x_mean
    return a, b


def predict_price(test_data: DataFrame, train_data: DataFrame, regression_params: List[Tuple[float, float]]) -> ndarray:
    x_test = (test_data.index - train_data.index[0]).days
    y_pred = np.zeros_like(x_test, dtype=float)
    for segment_a, segment_b in regression_params:
        segment_pred = segment_a * x_test + segment_b
        y_pred += np.where(x_test >= 0, segment_pred, 0)  # Unikanie ujemnych prognoz
    return y_pred


class SegmentedRegression:

    def __init__(self, train_data: DataFrame):
        self.train_data = train_data
        self.regression_params = get_segmented_regression(self.train_data)

    def predict(self, test_data: DataFrame):
        return predict_price(test_data, self.train_data, self.regression_params)
