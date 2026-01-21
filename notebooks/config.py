import os
import logging

from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
)

SEED = 42
DATA_DIR = Path('data/processed')

os.chdir(Path('..').resolve())