import argparse
from enum import Enum


class TimeFrameAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            time_frame = TimeFrame(values.lower())
            setattr(namespace, self.dest, time_frame)
        except ValueError:
            parser.error("Invalid time frame provided")


class TimeFrame(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

    def __str__(self):
        return self.value
