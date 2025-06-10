from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, HttpUrl, Field


class ContentType(str, Enum):
    """Content types for search requests."""
    IMAGES = "images"
    ARTICLES = "articles"
    NEWS = "news"
    WEBPAGE = "webpage"


class SearchAction(str, Enum):
    """Actions for search requests."""
    SEARCH_IMAGES = "search_images"
    DOWNLOAD_ARTICLES = "download_articles"
    DOWNLOAD_WEBPAGE = "download_webpage"


class DownloadStatus(str, Enum):
    """Download result statuses."""
    SUCCESS = "success"
    FAILED = "failed"


class SearchRequest(BaseModel):
    """User's parsed search request."""
    action: SearchAction
    subject: str
    count: int = Field(gt=0, description="Number of items to download")
    content_type: ContentType
    source: str = "any"  # Website source or 'any'
    topic: str = "random"  # Specific topic or 'random'


class SearchCandidate(BaseModel):
    """A candidate result from search engine."""
    url: HttpUrl
    title: str
    description: str | None = None
    score: float = 0.0  # Relevance score
    engine: str | None = None  # Which search engine found this


class DownloadResult(BaseModel):
    """Result of downloading a file."""
    url: HttpUrl
    filename: str
    filepath: Path
    size_bytes: int
    download_time: datetime = Field(default_factory=datetime.now)
    status: DownloadStatus
    error_message: str | None = None
    mime_type: str | None = None

    # Article-specific fields
    title: str | None = None
    extracted_text: str | None = None

    # Image-specific fields
    image_format: str | None = None
    image_dimensions: tuple[int, int] | None = None
    is_valid_image: bool = True


class ImageSearchResult(BaseModel):
    """Image search result with validation."""
    url: HttpUrl
    title: str | None = None
    thumbnail_url: HttpUrl | None = None
    size_estimate: str | None = None
    source_site: str | None = None
    score: float = 0.0
