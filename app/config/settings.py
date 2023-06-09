from pathlib import Path

from core.utils.enums import TimeFrame

BASE_DIR = Path(__file__).resolve().parent.parent


# DATASET PATH CONFIG
DATASETS = {
    TimeFrame.DAILY: BASE_DIR / "datasets/usdpln-daily.csv",
    TimeFrame.WEEKLY: BASE_DIR / "datasets/usdpln-weekly.csv",
    TimeFrame.MONTHLY: BASE_DIR / "datasets/usdpln-monthly.csv",
    TimeFrame.QUARTERLY: BASE_DIR / "datasets/usdpln-quarterly.csv",
    TimeFrame.YEARLY: BASE_DIR / "datasets/usdpln-yearly.csv",
}

CHARTS_DIR = BASE_DIR / "charts"
