from typing import List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from core.models.linear_regression import LinearRegression


def predict_price(data: DataFrame, segments: List) -> Tuple[DataFrame, DataFrame]:
    segment_dates = []
    segment_actuals = []
    segment_prediction_dates = []
    segment_predictions = []
    for segment in segments:
        segment_data = data.iloc[segment]
        train_data = segment_data.iloc[:-1]

        # Check if the segment has more than one row
        if len(train_data) <= 0:
            continue

        test_data = segment_data.iloc[-1:]
        model = LinearRegression(train_data)
        y_pred = model.predict(test_data)
        segment_prediction_dates.extend(test_data.index)
        segment_dates.extend(segment_data.index)
        segment_actuals.extend(segment_data['Close'])
        segment_predictions.append(y_pred[0])

    segment_prediction_dates = [date for date, prediction in zip(segment_prediction_dates, segment_predictions) if
                                not np.isnan(prediction)]
    segment_predictions = [prediction for prediction in segment_predictions if not np.isnan(prediction)]
    actual_data_dict = {
        'Date': segment_dates,
        'Close': segment_actuals
    }
    predictions_data_dict = {
        'Date': segment_prediction_dates,
        'Close': segment_predictions
    }
    print(actual_data_dict)
    print(len(actual_data_dict["Date"]))
    print(len(actual_data_dict["Close"]))

    # Create a dataframe
    actual_data = pd.DataFrame(actual_data_dict, index=actual_data_dict["Date"])
    predicted_data = pd.DataFrame(predictions_data_dict, index=predictions_data_dict["Date"])
    return actual_data, predicted_data


def get_segments(data: DataFrame) -> List:
    segments = []
    current_segment = [0]
    for i in range(1, len(data)):
        if data['Close'][i] < data['Close'][i - 1]:
            segments.append(current_segment)
            current_segment = [i]
        else:
            current_segment.append(i)
    segments.append(current_segment)
    return segments


class SegmentedRegression:

    def __init__(self, data: DataFrame):
        self.data = data
        self.segments = get_segments(data)

    def predict(self):
        return predict_price(self.data, self.segments)
