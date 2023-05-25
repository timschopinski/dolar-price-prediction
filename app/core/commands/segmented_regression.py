import logging
from argparse import Namespace
from typing import List
from core.data.visualization import inspect_data
from core.management import BaseCommand, BaseCommandArgumentParser
from core.data.data_extractor import get_data
from matplotlib import pyplot as plt
from core.models.linear_regression import LinearRegression
from core.utils.files import save_chart
import numpy as np


class Command(BaseCommand):
    def __init__(self):
        super().__init__()

    def handle(self, *args, **kwargs):
        args = self.get_parsed_args()
        data = get_data(args.time_frame, args.date_from, args.date_to)
        inspect_data(data, self.logger, "DATA")

        segments = []
        current_segment = [0]
        for i in range(1, len(data)):
            if data['Close'][i] < data['Close'][i - 1]:
                segments.append(current_segment)
                current_segment = [i]
            else:
                current_segment.append(i)
        segments.append(current_segment)

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

            test_data = segment_data.iloc[-1:]  # Use the last row for testing
            model = LinearRegression(train_data)
            y_pred = model.predict(test_data)
            segment_prediction_dates.extend(test_data.index)
            segment_dates.extend(segment_data.index)
            segment_actuals.extend(segment_data['Close'])
            segment_predictions.append(y_pred[0])

        segment_prediction_dates = [date for date, prediction in zip(segment_prediction_dates, segment_predictions) if
                                    not np.isnan(prediction)]
        segment_predictions = [prediction for prediction in segment_predictions if not np.isnan(prediction)]

        self.plot(segment_dates, segment_actuals, segment_prediction_dates, segment_predictions, args)

    def plot(
            self,
            segment_dates: List,
            segment_actuals: List,
            segment_prediction_dates_num: List,
            segment_predictions: List,
            args: Namespace
    ):
        plt.figure(figsize=(12, 6))
        plt.plot(segment_dates, segment_actuals, marker='o', label='Actual', c='#1f77b4')
        plt.plot(segment_prediction_dates_num, segment_predictions, marker='o', label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Segmented Linear Regression')
        plt.legend()
        plt.xticks(rotation=45)
        save_chart(args.title, self.logger)
        plt.show()

    def get_parsed_args(self) -> Namespace:
        parser = BaseCommandArgumentParser(description='Calculate price using Linear Regression and save chart')
        args, _ = parser.parse_known_args(title="segmented-regression")
        if args.verbose:
            self.logger.setLevel(logging.INFO)
        return args
