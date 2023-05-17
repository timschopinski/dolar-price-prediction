import argparse
from argparse import Namespace

from core.management import BaseCommand
from matplotlib import pyplot as plt
from data.data_extractor import get_data
from utils.enums import TimeFrame, TimeFrameAction


class Command(BaseCommand):

    def handle(self, argv=None):
        args = self.get_parsed_args()
        data = get_data(args.time_frame, args.date_from, args.date_to)
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.set_facecolor("white")
        fig.set_facecolor("white")
        plt.plot(data["Close"], lw=0.5, label="BTC")
        plt.xlabel("Data")
        plt.ylabel("Cena")
        plt.title("USD/PLN")
        plt.legend(loc="upper left")
        plt.savefig(f"charts/usd-pln-{args.time_frame}.png")
        plt.show()
        self.logger.info("Chart saved successfully")

    @staticmethod
    def get_parsed_args() -> Namespace:
        parser = argparse.ArgumentParser(description='Extract data within a specified date range.')
        parser.add_argument('--time_frame', type=str, help='Time frame for data extraction',
                            default=TimeFrame.DAILY, choices=[str(time_frame) for time_frame in TimeFrame],
                            action=TimeFrameAction)
        parser.add_argument('--date_from', type=str, help='Start date for data extraction (YYYY-MM-DD)', default=None)
        parser.add_argument('--date_to', type=str, help='End date for data extraction (YYYY-MM-DD)', default=None)
        args, _ = parser.parse_known_args()
        return args
