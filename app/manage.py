import sys

from core.management import CommandHandler


def main():
    """Run administrative tasks."""
    command_handler = CommandHandler(sys.argv)
    command_handler.execute_from_command_line()


if __name__ == '__main__':
    main()
