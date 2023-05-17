import unittest
from data.data_extractor import get_data
from utils.enums import TimeFrame
from pandas import DataFrame, DatetimeIndex


class DataExtractorTest(unittest.TestCase):
    def setUp(self):
        self.data: DataFrame = get_data(TimeFrame.DAILY)

    def test_get_data_columns(self):
        expected_columns = [
            "Open",
            "High",
            "Low",
            "Close",
        ]
        assert list(self.data.columns) == expected_columns

    def test_get_data_index_is_date(self):
        self.assertIsInstance(self.data.index, DatetimeIndex)
