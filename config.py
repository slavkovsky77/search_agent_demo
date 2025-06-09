import logging
import colorlog


# Configure colored logging
def setup_logging():
    """Set up colored logging for console and plain logging for file."""

    # Create logger
    logger = logging.getLogger(__name__)
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

    # File handler with plain text
    # file_handler = logging.FileHandler('agent_search.log')
    # file_handler.setLevel(logging.DEBUG)

    # Plain formatter for file
    # file_formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # )
    # file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger
