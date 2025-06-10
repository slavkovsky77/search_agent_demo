"""
File downloader with validation and metadata extraction.
"""
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO

from .models import DownloadResult, ImageSearchResult, SearchCandidate
from .config import setup_logging

logger = setup_logging(__name__)


class FileDownloader:
    """Handles file downloads with validation and metadata extraction."""

    def __init__(self, download_dir: Path):
        self.download_dir = download_dir
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        })

    def download_image(self, image_result: ImageSearchResult, directory: Path) -> DownloadResult:
        """Download and validate an image file."""
        logger.info(f"📷 Downloading image: {image_result.url}")

        headers = {
            **self.session.headers,
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Referer': 'https://www.google.com/'
        }

        try:
            response = requests.get(str(image_result.url), timeout=30, headers=headers, stream=True)
            response.raise_for_status()

            content = response.content

            # Validate minimum size
            if len(content) < 1000:
                raise Exception(f"Image too small: {len(content)} bytes")

            # Validate it's actually an image using PIL
            image_info = self._validate_image_content(content)
            if not image_info['is_valid']:
                raise Exception(f"Invalid image content: {image_info['error']}")

            # Generate filename
            filename = self._generate_filename(
                image_result.url,
                response.headers.get('content-type'),
                prefix="image"
            )

            # Save file
            filepath = directory / filename
            filepath.write_bytes(content)

            return DownloadResult(
                url=image_result.url,
                filename=filename,
                filepath=filepath,
                size_bytes=len(content),
                status="success",
                mime_type=response.headers.get('content-type'),
                image_format=image_info['format'],
                image_dimensions=image_info['dimensions'],
                is_valid_image=True
            )

        except Exception as e:
            logger.error(f"❌ Image download failed {image_result.url}: {e}")
            return DownloadResult(
                url=image_result.url,
                filename="failed_download",
                filepath=Path("failed"),
                size_bytes=0,
                status="failed",
                error_message=str(e),
                is_valid_image=False
            )

    def download_webpage(self, candidate: SearchCandidate, directory: Path) -> DownloadResult:
        """Download a webpage/article."""
        logger.info(f"📄 Downloading webpage: {candidate.url}")

        try:
            response = self.session.get(str(candidate.url), timeout=15)
            response.raise_for_status()

            # Generate clean filename from title
            safe_title = re.sub(r'[^\w\-_.]', '_', candidate.title)
            filename = f"{safe_title}.html"
            filepath = directory / filename

            # Save HTML content
            filepath.write_text(response.text, encoding='utf-8')

            return DownloadResult(
                url=candidate.url,
                filename=filename,
                filepath=filepath,
                size_bytes=len(response.text.encode('utf-8')),
                status="success",
                mime_type=response.headers.get('content-type'),
                title=candidate.title
            )

        except Exception as e:
            logger.error(f"❌ Webpage download failed {candidate.url}: {e}")
            return DownloadResult(
                url=candidate.url,
                filename="failed_download",
                filepath=Path("failed"),
                size_bytes=0,
                status="failed",
                error_message=str(e),
                title=candidate.title
            )

    def _validate_image_content(self, content: bytes) -> dict:
        """Validate that content is actually a valid image using PIL."""
        try:
            with Image.open(BytesIO(content)) as img:
                return {
                    'is_valid': True,
                    'format': img.format,
                    'dimensions': img.size,
                    'error': None
                }
        except Exception as e:
            return {
                'is_valid': False,
                'format': None,
                'dimensions': None,
                'error': str(e)
            }

    def _generate_filename(self, url: str, content_type: str | None = None,
                           prefix: str = "download") -> str:
        """Generate a proper filename from URL and content type."""
        try:
            # Try to extract filename from URL
            parsed_url = urlparse(str(url))
            filename = Path(unquote(parsed_url.path)).name

            if filename and '.' in filename:
                return filename

        except Exception:
            pass

        # Generate filename from content type or use default
        if content_type:
            ext = mimetypes.guess_extension(content_type) or '.bin'
        else:
            ext = '.jpg' if prefix == "image" else '.html'

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{prefix}_{timestamp}{ext}"
