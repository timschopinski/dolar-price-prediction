import os
import pkgutil
import subprocess

from config.settings import BASE_DIR
import logging
from importlib import import_module
from abc import ABC, abstractmethod


class BaseCommand(ABC):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    def handle(self, *args, **kwargs):
        raise NotImplementedError('subclasses of BaseCommand must provide a handle() method')


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
                module = import_module(f".{commands[0]}", "core.commands")
                module.Command().handle(self.argv)
            except AttributeError:
                self.logger.error("Failed to execute Command.")
