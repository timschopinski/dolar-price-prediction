import os
import logging
from pathlib import Path
from matplotlib import pyplot as plt
from config.settings import CHARTS_DIR


def save_chart(title: str, logger: logging.Logger):
    path = Path(CHARTS_DIR, f"{title}.png")
    try:
        plt.savefig(path)
        logger.info("Chart saved successfully")
    except FileNotFoundError:
        logger.warning("The 'charts' directory does not exist. Create the directory and try again.")
        os.makedirs(CHARTS_DIR.name)
        plt.savefig(path)
        logger.info("Chart saved successfully")
