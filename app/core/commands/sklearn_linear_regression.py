import logging
from argparse import Namespace
from pandas import DataFrame
from core.data.visualization import inspect_data
from core.management import BaseCommand, BaseCommandArgumentParser
from core.data.mse import calculate_mse
from core.data.data_extractor import get_data, split
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from core.utils.files import save_chart


class Command(BaseCommand):
    def __init__(self):
        super().__init__()

    def handle(self, *args, **kwargs):
        args = self.get_parsed_args()
        data = get_data(args.time_frame, args.date_from, args.date_to)

        train_data, test_data = split(data, args.test_size)
        inspect_data(train_data, self.logger, "Train Data")
        inspect_data(test_data, self.logger, "Test Data")

        model = LinearRegression()
        x_train = (train_data.index - train_data.index[0]).days.values.reshape(-1, 1)
        y_train = train_data['Close'].values
        model.fit(x_train, y_train)

        x_test = (test_data.index - train_data.index[0]).days.values.reshape(-1, 1)
        y_pred = model.predict(x_test)
        mse = calculate_mse(y_pred, test_data['Close'].values)
        self.logger.info(f'MSE: {mse}')

        self.plot(test_data, y_pred, args)

    def plot(self, test_data: DataFrame, y_pred: DataFrame, args: Namespace):
        plt.plot(test_data.index, y_pred, label='Regression Line')
        plt.scatter(test_data.index, test_data['Close'], s=2, label='Actual Data')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.title("SciKitLearn Linear Regression")
        save_chart(args.title, self.logger)
        plt.show()

    def get_parsed_args(self) -> Namespace:
        parser = BaseCommandArgumentParser(
            description='Calculate price using Linear Regression with sklearn and save chart'
        )
        parser.add_argument('--test_size', type=float, help='Test data size', default=0.2)
        args, _ = parser.parse_known_args(title="sklearn-linear-regression")
        if args.verbose:
            self.logger.setLevel(logging.INFO)
        return args
