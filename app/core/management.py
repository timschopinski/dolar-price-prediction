import os
import pkgutil
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from config.settings import BASE_DIR
import logging
from importlib import import_module
from abc import ABC, abstractmethod
from core.utils.enums import TimeFrame, TimeFrameAction


class BaseCommand(ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        stream_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stream_handler)

    @abstractmethod
    def handle(self, *args, **kwargs):
        raise NotImplementedError('subclasses of BaseCommand must provide a handle() method')


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
