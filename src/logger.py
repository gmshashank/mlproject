import logging
import os
from datetime import datetime


LOGS_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOGS_FILE)

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOGS_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)s %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
