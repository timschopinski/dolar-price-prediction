from typing import Tuple

import pandas as pd
from pandas import DataFrame
from utils.enums import TimeFrame
from pathlib import Path
from config.settings import DATASETS
from matplotlib import pyplot as plt


def get_dataset(time_frame: TimeFrame) -> Path:
    return DATASETS[time_frame]


def get_data(
    time_frame: TimeFrame,
    date_from: str | None = None,
    date_to: str | None = None,
    slice_: int | None = None,
    reverse: bool = False,
) -> DataFrame:

    data = pd.read_csv(
        get_dataset(time_frame), parse_dates=["Date"], index_col=["Date"]
    )
    data.dropna(inplace=True)
    if date_from:
        mask = data.index >= date_from
        data = data.loc[mask]
    if date_to:
        mask = data.index <= date_to
        data = data.loc[mask]
    if slice_:
        data = data.head(slice_)
    if reverse:
        data.sort_index(ascending=False, inplace=True)

    return data


def split(data: DataFrame, test_size: float = 0.2) -> Tuple:
    n_test = int(len(data) * test_size)
    train_data, test_data = data.iloc[n_test:], data.iloc[:n_test]
    return train_data, test_data
