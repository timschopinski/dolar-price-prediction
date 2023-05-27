from core.data.visualization import inspect_data, inspect_predictions
from core.management import RegressionCommand
from core.data.data_extractor import get_data, split
from core.models.linear_regression import LinearRegression


class Command(RegressionCommand):
    def __init__(self):
        super().__init__()

    def handle(self, *args, **kwargs):
        args = self.get_parsed_args()
        data = get_data(args.time_frame, args.date_from, args.date_to)
        train_data, test_data = split(data, args.test_size)

        inspect_data(train_data, self.logger, "Train Data")
        inspect_data(test_data, self.logger, "Test Data")

        model = LinearRegression(train_data)
        y_pred = model.predict(test_data)

        inspect_predictions(test_data['Close'].values, y_pred, self.logger)
        self.plot(test_data, y_pred, args.title)
