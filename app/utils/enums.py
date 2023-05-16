from enum import Enum


class TimeFrame(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

    def __str__(self):
        return self.value
