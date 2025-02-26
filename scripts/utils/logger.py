import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(log_file_name, log_dir=None):
    """ Sets up a logger that writes different log levels to separate files. """
    if not log_dir:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'logs')
        # log_dir = Path("logs")

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    # log_dir = Path("logs")
    # log_dir.mkdir(exist_ok=True)

    # Define log format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Capture all levels
    # logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers for different log levels
    # info_handler = logging.FileHandler(os.path.join(log_dir, f"{log_file_name}_info.log"), encoding="utf-8")
    info_handler = RotatingFileHandler(Path(log_dir, f"{log_file_name}_info.log"), maxBytes=10000, backupCount=1, encoding="utf-8")
    info_handler.setLevel(logging.INFO)

    # warning_handler = logging.FileHandler(os.path.join(log_dir, f"{log_file_name}_warning.log"), encoding="utf-8")
    warning_handler = RotatingFileHandler(Path(log_dir, f"{log_file_name}_warning.log"), maxBytes=10000, backupCount=1, encoding="utf-8")
    warning_handler.setLevel(logging.WARNING)

    # error_handler = logging.FileHandler(os.path.join(log_dir, f"{log_file_name}_error.log"), encoding="utf-8")
    error_handler = RotatingFileHandler(Path(log_dir, f"{log_file_name}_error.log"), maxBytes=10000, backupCount=1, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)

    # Console  handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # Define formatter with emojis
    class EmojiFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.DEBUG:
                record.levelname = "[🐛 DEBUG]"
            elif record.levelno == logging.INFO:
                record.levelname = "[✅ INFO]"
            elif record.levelno == logging.WARNING:
                record.levelname = "[⚠️ WARNING]"
            elif record.levelno == logging.ERROR:
                record.levelname = "[❌ ERROR]"
            elif record.levelno == logging.CRITICAL:
                record.levelname = "[🚨 CRITICAL]"
            return super().format(record)
        
    # Define formatter and Apply formatter with emojis to handlers
    formatter = EmojiFormatter(fmt=log_format, datefmt=date_format)#, encoding='utf-8')

    # Apply formatter to handlers
    for handler in [info_handler, warning_handler, error_handler, console_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger