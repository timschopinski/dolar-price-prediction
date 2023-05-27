from matplotlib import pyplot as plt
from logging import Logger
import pandas as pd
import numpy as np

from core.data.error_metrics import calculate_mse, calculate_rmse, calculate_mae


def print_title(title: str, logger: Logger) -> None:
    logger.info("*" * 40)
    title_break = ' ' * int((38 - len(title)) / 2)
    logger.info(f"*{title_break}{title}{title_break}*")
    logger.info("*" * 40)


def inspect_data(dataset: pd.DataFrame, logger: Logger, title: str) -> None:
    print_title(title, logger)
    logger.info('Dataset shape:')
    logger.info(dataset.shape)
    logger.info("-" * 40)

    logger.info('Missing Values:')
    logger.info(dataset.isnull().sum())
    logger.info("-" * 40)

    logger.info('Data Types:')
    logger.info(dataset.dtypes)
    logger.info("-" * 40)

    logger.info('Tail:')
    logger.info(dataset.tail())
    logger.info("-" * 40)

    logger.info('Statistics:')
    logger.info(dataset.describe().transpose())
    logger.info("-" * 40)
    plt.show()
    logger.info("")


def inspect_predictions(actual_data: np.ndarray, predictions: np.ndarray, logger: Logger) -> None:
    print_title("PREDICTION ANALYSIS", logger)
    mse = calculate_mse(predictions, actual_data)
    logger.info("-" * 40)
    logger.info(f"MSE: {mse}")

    mae = calculate_mae(predictions, actual_data)
    logger.info("-" * 40)
    logger.info(f"MAE: {mae}")

    rmse = calculate_rmse(predictions, actual_data)
    logger.info("-" * 40)
    logger.info(f"RMSE: {rmse}")
    logger.info("-" * 40)
    logger.info("")
