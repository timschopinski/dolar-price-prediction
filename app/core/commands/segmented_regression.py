import logging
from argparse import Namespace
from pandas import DataFrame
from core.data.visualization import inspect_data
from core.management import BaseCommand, BaseCommandArgumentParser
from core.data.data_extractor import get_data
from matplotlib import pyplot as plt
from core.models.segmented_regression import SegmentedRegression
from core.utils.files import save_chart


class Command(BaseCommand):
    def __init__(self):
        super().__init__()

    def handle(self, *args, **kwargs):
        args = self.get_parsed_args()
        data = get_data(args.time_frame, args.date_from, args.date_to)
        inspect_data(data, self.logger, "DATA")
        model = SegmentedRegression(data)

        actual_data, predicted_data = model.predict()
        self.plot(actual_data, predicted_data, args)

    def plot(self, actual_data: DataFrame, predicted_data: DataFrame, args: Namespace):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_data.index, actual_data['Close'], marker='o', label='Actual', c='#1f77b4')
        plt.plot(predicted_data.index, predicted_data['Close'], marker='o', label='Predicted', color='orange')
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
