from ann_visualizer.visualize import ann_viz

from core.data.visualization import inspect_data, inspect_predictions
from core.management import NeuralNetworkCommand
from core.data.data_extractor import get_data, split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from core.models.neural_network import NeuralNetwork, Layer


class Command(NeuralNetworkCommand):
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
        train_target = train_target.reshape(-1, 1)  # Reshape train_target

        model = NeuralNetwork()
        model.add(Layer(train_features.shape[1], 8, activation='relu'))
        model.add(Layer(8, 1, activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        losses = model.fit(train_features, train_target, epochs=args.epochs, batch_size=16)
        test_features = test_data_scaled[:, :-1]
        predictions = model.predict(test_features)
        predictions = scaler.inverse_transform(np.concatenate((test_features, predictions), axis=1))[:, -1]

        inspect_predictions(test_data['Close'].values, predictions, self.logger)
        self.plot(test_data, predictions, losses, args.title)
