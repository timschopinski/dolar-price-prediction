from matplotlib import pyplot as plt
from logging import Logger
from pandas import DataFrame


def inspect_data(dataset: DataFrame, logger: Logger, title: str):
    logger.info("*" * 40)
    title_break = ' ' * int((38 - len(title)) / 2)
    logger.info(f"*{title_break}{title}{title_break}*")
    logger.info("*" * 40)

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
