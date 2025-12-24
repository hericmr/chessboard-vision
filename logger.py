"""
Chess Vision Logger

Centralized logging for debugging move detection, noise, and API calls.
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str = "chess_vision", log_file: bool = True) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Whether to create log file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler (INFO level)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter('%(message)s')
    console.setFormatter(console_fmt)
    logger.addHandler(console)
    
    # File handler (DEBUG level)
    if log_file:
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(log_dir, "chess_vision.log")
        
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(message)s',
            datefmt='%H:%M:%S'
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
_logger = None

def get_logger() -> logging.Logger:
    """Get or create the global logger."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def log_move(move_uci: str, status: str, source: str = "vision"):
    """Log a detected or received move."""
    logger = get_logger()
    logger.info(f"[{source.upper()}] MOVE: {move_uci} ({status})")
    logger.debug(f"Move details: uci={move_uci}, status={status}, source={source}")


def log_noise(state: str, changed_count: int):
    """Log noise detection event."""
    logger = get_logger()
    logger.debug(f"[NOISE] State: {state}, Changed: {changed_count}")


def log_api(action: str, result: str, details: str = ""):
    """Log API call."""
    logger = get_logger()
    logger.debug(f"[API] {action}: {result} {details}")


def log_error(message: str, exc: Exception = None):
    """Log an error."""
    logger = get_logger()
    if exc:
        logger.error(f"[ERROR] {message}: {exc}")
    else:
        logger.error(f"[ERROR] {message}")


def log_session_start():
    """Log session start marker."""
    logger = get_logger()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n{'='*50}")
    logger.info(f"  Session started: {timestamp}")
    logger.info(f"{'='*50}")
