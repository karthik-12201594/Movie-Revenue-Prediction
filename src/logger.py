import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    # file handler
    log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
