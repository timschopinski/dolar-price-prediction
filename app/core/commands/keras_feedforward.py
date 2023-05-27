import logging
from argparse import Namespace
from pandas import DataFrame
from core.data.visualization import inspect_data, inspect_predictions
from core.management import BaseCommand, BaseCommandArgumentParser
from core.data.data_extractor import get_data, split
from matplotlib import pyplot as plt
from core.utils.files import save_chart
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


class Command(BaseCommand):
    def __init__(self):
        super().__init__()

    def handle(self, *args, **kwargs):
        args = self.get_parsed_args()
        data = get_data(args.time_frame, args.date_from, args.date_to)

        train_data, test_data = split(data, args.test_size)
        inspect_data(train_data, self.logger, "Train Data")
        inspect_data(test_data, self.logger, "Test Data")

        scaler = MinMaxScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        test_data_scaled = scaler.transform(test_data)

        train_features = train_data_scaled[:, :-1]
        train_target = train_data_scaled[:, -1]

        model = keras.Sequential([
            keras.layers.Dense(8, activation='relu', input_shape=(train_features.shape[1],)),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(train_features, train_target, epochs=1000, batch_size=16)
        test_features = test_data_scaled[:, :-1]
        predictions = model.predict(test_features)
        predictions = scaler.inverse_transform(np.concatenate((test_features, predictions), axis=1))[:, -1]
        inspect_predictions(test_data['Close'].values, predictions, self.logger)
        self.plot(test_data, predictions, args)

    def plot(self, actual_values: DataFrame, predictions: np.ndarray, args: Namespace):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_values.index, actual_values['Close'].values, marker='o', label='Actual', c='#1f77b4')
        plt.plot(actual_values.index, predictions, marker='o', label='Predicted', color='orange')
        plt.title('USD/PLN Actual vs. Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title("Keras FeedForward Neural Networks")
        plt.legend()
        save_chart(args.title, self.logger)
        plt.show()

    def get_parsed_args(self) -> Namespace:
        parser = BaseCommandArgumentParser(
            description='Calculate price using Feedforward Neural Networks with keras'
        )
        parser.add_argument('--test_size', type=float, help='Test data size', default=0.2)
        args, _ = parser.parse_known_args(title="keras-feedforward-neural-network")
        if args.verbose:
            self.logger.setLevel(logging.INFO)
        return args
