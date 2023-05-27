import os
import pkgutil
import subprocess
import sys
from argparse import ArgumentParser
from config.settings import BASE_DIR
from importlib import import_module
from abc import ABC, abstractmethod
from core.utils.enums import TimeFrame, TimeFrameAction
import logging
from argparse import Namespace
from typing import List
from pandas import DataFrame
from matplotlib import pyplot as plt
from core.utils.files import save_chart
import numpy as np
import inspect


class BaseCommandArgumentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--time_frame', type=str, help='Time frame for data extraction',
                          default=TimeFrame.DAILY, choices=[str(time_frame) for time_frame in TimeFrame],
                          action=TimeFrameAction)
        self.add_argument('--date_from', type=str, help='Start date for data extraction (YYYY-MM-DD)', default=None)
        self.add_argument('--date_to', type=str, help='End date for data extraction (YYYY-MM-DD)', default=None)
        self.add_argument('--title', type=str, help='Chart title', default=None)
        self.add_argument('--verbose', type=bool, help='Enable verbose logging (True/False)', default=False)

    def parse_known_args(
        self, args=None, namespace=None, **kwargs
    ) -> tuple[Namespace, list[str]]:
        self.set_defaults(title=kwargs.get("title", ""))
        return super().parse_known_args(args, namespace)


class BaseCommand(ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        stream_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stream_handler)

    @abstractmethod
    def handle(self, *args, **kwargs):
        raise NotImplementedError('subclasses of BaseCommand must provide a handle() method')

    @staticmethod
    def plot_predictions(actual_values: DataFrame, predictions: np.ndarray, title: str, logger: logging.Logger):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_values.index, actual_values['Close'].values, marker='o', label='Actual', c='#1f77b4')
        plt.plot(actual_values.index, predictions, marker='o', label='Predicted', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(title.replace("-", " ").capitalize())
        plt.legend()
        save_chart(title, logger)
        plt.show()


class RegressionCommand(BaseCommand):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle(self, *args, **kwargs):
        super().handle(*args, **kwargs)

    def plot(self, actual_values: DataFrame, predictions: np.ndarray, title: str):
        self.plot_predictions(actual_values, predictions, title, self.logger)

    def get_parsed_args(self) -> Namespace:
        parser = BaseCommandArgumentParser(
            description='Calculate price using Linear Regression with sklearn and save chart'
        )
        parser.add_argument('--test_size', type=float, help='Test data size', default=0.2)
        frame = inspect.currentframe().f_back
        module_name = inspect.getmodule(frame).__name__
        title = module_name.split('.')[-1].replace("_", "-")
        args, _ = parser.parse_known_args(title=title)
        if args.verbose:
            self.logger.setLevel(logging.INFO)
        return args


class NeuralNetworkCommand(BaseCommand, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle(self, *args, **kwargs):
        super().handle(*args, **kwargs)

    def plot(self, actual_values: DataFrame, predictions: np.ndarray, losses: List[np.ndarray], title: str):
        self.plot_predictions(actual_values, predictions, title, self.logger)
        self.plot_losses(losses, title, self.logger)

    @staticmethod
    def plot_losses(losses: List[np.ndarray], title: str, logger: logging.Logger):
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.title("")
        plt.legend("FeedForward Neural Network Losses")
        save_chart(f"{title}-losses", logger)
        plt.show()

    def get_parsed_args(self) -> Namespace:
        parser = BaseCommandArgumentParser(
            description='Calculate price using Feedforward Neural Networks'
        )
        parser.add_argument('--test_size', type=float, help='Test data size. Default = 0.2', default=0.2)
        parser.add_argument('--epochs', type=int, help='Number of epochs. Default = 1000', default=1000)
        frame = inspect.currentframe().f_back
        module_name = inspect.getmodule(frame).__name__
        title = module_name.split('.')[-1].replace("_", "-")
        args, _ = parser.parse_known_args(title=title)
        if args.verbose:
            self.logger.setLevel(logging.INFO)
        return args


class CommandHandler:
    def __init__(self, argv=None):
        self.argv = argv
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    @staticmethod
    def find_commands():
        command_dir = os.path.join(BASE_DIR, 'core', 'commands')
        return [name for _, name, is_pkg in pkgutil.iter_modules([command_dir])
                if not is_pkg and not name.startswith('_')]

    def execute_from_command_line(self):
        commands = self.find_commands()
        if len(commands) == 0:
            self.logger.error("No commands implemented.")

        if self.argv[1] == "test":
            subprocess.run(["python", "-m", "unittest"])
            return

        if self.argv[1] not in commands:
            self.logger.error("Invalid command.")
        else:
            try:
                command = commands[commands.index(self.argv[1])]
                module = import_module(f".{command}", "core.commands")
                module.Command().handle(self.argv)
            except AttributeError as e:
                self.logger.error("Failed to execute Command.", e)
