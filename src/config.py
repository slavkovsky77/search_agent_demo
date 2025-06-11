import logging
import colorlog


# Configure colored logging
def setup_logging(name):
    """Set up colored logging for console and plain logging for file."""

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Colored formatter for console
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(color_formatter)

    logger.addHandler(console_handler)

    return logger
