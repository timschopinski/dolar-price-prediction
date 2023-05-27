import pandas as pd
from core.data.visualization import inspect_data, inspect_predictions
from core.management import RegressionCommand
from core.data.data_extractor import get_data
from matplotlib import pyplot as plt
from core.models.segmented_regression import SegmentedRegression
from core.utils.files import save_chart


class Command(RegressionCommand):
    def __init__(self):
        super().__init__()

    def handle(self, *args, **kwargs):
        args = self.get_parsed_args()
        data = get_data(args.time_frame, args.date_from, args.date_to)
        inspect_data(data, self.logger, "DATA")
        model = SegmentedRegression(data)

        actual_data, predicted_data = model.predict()
        merged_df = pd.merge(actual_data, predicted_data, on='Date', how='inner')
        actual_values = merged_df['Close_x']
        predicted_values = merged_df['Close_y']

        inspect_predictions(actual_values, predicted_values, self.logger)
        self.plot(actual_data, predicted_data, args.title)

    def plot(self, actual_data: pd.DataFrame, predicted_data: pd.DataFrame, title: str):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_data.index, actual_data['Close'], marker='o', label='Actual', c='#1f77b4')
        plt.plot(predicted_data.index, predicted_data['Close'], marker='o', label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title('Segmented Linear Regression')
        plt.legend()
        plt.xticks(rotation=45)
        save_chart(title, self.logger)
        plt.show()
