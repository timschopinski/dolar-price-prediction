from argparse import Namespace
from matplotlib import pyplot as plt
from core.data.data_extractor import get_data
from core.management import BaseCommand, BaseCommandArgumentParser
from core.utils.files import save_chart


class Command(BaseCommand):

    def handle(self, *args, **kwargs):
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
        save_chart(args.title, self.logger)
        plt.show()

    @staticmethod
    def get_parsed_args() -> Namespace:
        parser = BaseCommandArgumentParser(description='Extract data within a specified date range.')
        args, _ = parser.parse_known_args(title="usd-pln")
        return args
