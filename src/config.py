import logging
import colorlog


# LLM Model Configuration
class LLMModels:
    """Centralized LLM model configuration for different use cases."""

    # Main models for different tasks
    REQUEST_UNDERSTANDING = "anthropic/claude-3-sonnet"
    QUERY_GENERATION = "anthropic/claude-3-sonnet"
    CONTENT_SCORING = "anthropic/claude-3-sonnet"
    IMAGE_SCORING = "anthropic/claude-3-sonnet"
    CONTENT_EXTRACTION = "anthropic/claude-3-haiku"  # Faster/cheaper for extraction
    VALIDATION = "anthropic/claude-3-sonnet"

    # Model parameters by task
    REQUEST_PARAMS = {"max_tokens": 200, "temperature": 0.1}
    QUERY_PARAMS = {"max_tokens": 200, "temperature": 0.3}
    SCORING_PARAMS = {"max_tokens": 500, "temperature": 0.1}
    IMAGE_SCORING_PARAMS = {"max_tokens": 100, "temperature": 0.1}
    EXTRACTION_PARAMS = {"max_tokens": 2000, "temperature": 0.1}
    VALIDATION_PARAMS = {"max_tokens": 200, "temperature": 0.1}


# System Constants
class SystemConstants:
    """Centralized system constants and configuration values."""

    # Search constants
    MAX_IMAGE_CANDIDATES = 20
    MAX_ARTICLE_CANDIDATES = 30
    SCORING_BATCH_SIZE = 16

    # Content extraction
    DEFAULT_MAX_TEXT_CHARS = 5000
    MIN_EXTRACTED_TEXT_LENGTH = 100
    MIN_ARTICLE_CONTENT_LENGTH = 200
    HTML_PREVIEW_CHARS = 15000

    # Image processing
    MAX_IMAGE_DIMENSION = 480  # Resize images to max 480p to save API costs
    IMAGE_QUALITY = 85

    # Network timeouts (seconds)
    IMAGE_DOWNLOAD_TIMEOUT = 30
    ARTICLE_DOWNLOAD_TIMEOUT = 15
    SEARCH_REQUEST_TIMEOUT = 10

    # Download multiplier calculation
    # Formula: 1 + (8 / (count + 1)) → 1→5x, 2→3x, 5→2x, 20→1.5x, 100→1.2x
    SMART_MULTIPLIER_FACTOR = 8

    # Default values
    DEFAULT_SEARXNG_URL = "http://localhost:8080"

    # User agent for web requests
    DEFAULT_USER_AGENT = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')


# Search Query Limits
class QueryLimits:
    """Limits for search query generation."""
    MAX_IMAGE_QUERIES = 5
    MAX_CONTENT_QUERIES = 3
    IMAGE_QUERY_WORDS_MIN = 1
    IMAGE_QUERY_WORDS_MAX = 4


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
