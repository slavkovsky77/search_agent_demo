"""
Search engine wrapper for SearXNG with candidate validation and scoring.
"""
import json
import base64
import requests
from PIL import Image
from io import BytesIO

from urllib.parse import urlparse
from openai import OpenAI
from pathlib import Path

from .models import SearchRequest, SearchCandidate, ImageSearchResult
from . import prompts
from .config import setup_logging
from .file_downloader import FileDownloader
from .content_extractor import ContentExtractor

logger = setup_logging(__name__)


class SearchEngine:
    """Handles search operations via SearXNG with intelligent candidate selection."""

    def __init__(self, searxng_url: str, openai_client: OpenAI):
        self.searxng_url = searxng_url.rstrip('/')
        self.client = openai_client
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        })

    def search_images(self, query: str, max_candidates: int = 20) -> list[ImageSearchResult]:
        """Search for images and return validated candidates."""
        logger.info(f"🔍 Searching images: '{query}' (up to {max_candidates} candidates)")

        search_url = f"{self.searxng_url}/search"
        params = {
            'q': f"{query} photograph",
            'categories': 'images',
            'format': 'json',
            'engines': 'bing images,google images,duckduckgo images'
        }

        try:
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            candidates = []
            for result in data.get('results', [])[:max_candidates]:
                if 'img_src' in result and self._is_valid_image_url(result['img_src']):
                    candidates.append(ImageSearchResult(
                        url=result['img_src'],
                        title=result.get('title'),
                        thumbnail_url=result.get('thumbnail_src'),
                        source_site=self._extract_domain(result['img_src'])
                    ))

            logger.info(f"🎯 Found {len(candidates)} valid image candidates")
            return candidates

        except Exception as e:
            logger.exception(f"❌ Image search failed for '{query}': {e}")
            return []

    def search_articles(self, query: str, max_candidates: int = 30) -> list[SearchCandidate]:
        """Search for articles and return candidates."""
        logger.info(
            f"🔍 Searching articles: '{query}' "
            f"(up to {max_candidates} candidates)"
        )

        search_url = f"{self.searxng_url}/search"
        # Always use 'news' category for articles - it finds actual articles, not homepages
        category = "news"
        engines = 'bing news,google news'

        params = {
            'q': query,
            'categories': category,
            'format': 'json',
            'engines': engines
        }

        try:
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            candidates = []
            for result in data.get('results', [])[:max_candidates]:
                if result.get('url'):
                    candidates.append(SearchCandidate(
                        url=result['url'],
                        title=result.get('title', 'Unknown Title'),
                        description=result.get('content', ''),
                        engine=result.get('engine', 'unknown')
                    ))

            logger.info(f"🎯 Found {len(candidates)} article candidates")
            return candidates

        except Exception as e:
            logger.exception(f"❌ Article search failed for '{query}': {e}")
            return []

    def generate_search_queries(self, search_request: SearchRequest) -> list[str]:
        """Generate search queries using AI."""
        if search_request.content_type == "images":
            prompt = prompts.get_search_queries_prompt(search_request.subject, search_request.count)
        else:
            prompt = prompts.get_content_queries_prompt(
                search_request.topic, search_request.source,
                search_request.content_type.value,
                search_request.count
            )

        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )

            queries = json.loads(response.choices[0].message.content.strip())
            if isinstance(queries, list):
                logger.info(f"🧠 Generated {len(queries)} search queries")
                return queries

        except Exception as e:
            logger.exception(f"❌ Query generation failed: {e}")

        # Fallback queries
        if search_request.content_type == "images":
            return [f"{search_request.subject} photograph", f"{search_request.subject} image"]
        else:
            return [search_request.subject]

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL looks like a valid image URL."""
        if not url or not url.startswith('http'):
            return False

        parsed = urlparse(url)
        path = parsed.path.lower()

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        has_image_ext = any(path.endswith(ext) for ext in image_extensions)

        has_image_keywords = any(
            keyword in url.lower()
            for keyword in ['image', 'photo', 'picture', 'img']
        )

        return has_image_ext or has_image_keywords

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"

    def score_and_select_candidates(self, candidates: list[SearchCandidate],
                                    search_request: SearchRequest,
                                    desired_count: int) -> list[SearchCandidate]:
        """Score candidates using AI and select the best ones (batch scoring with descriptions)."""
        if len(candidates) <= desired_count:
            return candidates

        logger.info(f"🎯 Scoring {len(candidates)} candidates to select best {desired_count}")

        # Prepare candidate descriptions for AI scoring
        candidate_descriptions = [
            f"{c.title} - {c.description[:100]}... ({c.url})"
            for c in candidates
        ]

        try:
            prompt = prompts.get_candidate_scoring_prompt(candidate_descriptions, search_request)

            response = self.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )

            scores = json.loads(response.choices[0].message.content.strip())

            # Assign scores to candidates
            for i, candidate in enumerate(candidates):
                if i < len(scores):
                    candidate.score = scores[i]
                else:
                    candidate.score = 0.0

            # Sort by score and return top candidates
            top_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)[:desired_count]

            logger.info(
                f"✅ Selected {len(top_candidates)} top candidates "
                f"(scores: {[f'{c.score:.2f}' for c in top_candidates]})"
            )
            return top_candidates

        except Exception as e:
            logger.exception(f"❌ Candidate scoring failed: {e}")
            # Fallback: return first N candidates
            return candidates[:desired_count]

    def score_candidate(self, candidate, content_type: str, search_request,
                        temp_dir: Path) -> float:
        """Score a single candidate (image or article) based on actual content."""
        try:
            if content_type == "images":
                return self._score_single_image(candidate, search_request.subject)
            else:
                return self._score_single_article(candidate, search_request, temp_dir)
        except Exception as e:
            logger.warning(f"⚠️ Scoring failed: {e}")
            return 0.0

    def _score_single_image(self, candidate, subject: str) -> float:
        """Score single image with vision API (resized to 480p max)."""

        try:
            # Download image
            response = requests.get(str(candidate.url), timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; SearchBot/1.0)'
            })
            response.raise_for_status()

            # Resize to max 480p to save API costs
            image = Image.open(BytesIO(response.content))
            if max(image.size) > 480:
                ratio = 480 / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to bytes
            img_buffer = BytesIO()
            image.save(img_buffer, format='JPEG', quality=85)
            image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            # Score with vision using prompts
            vision_prompt = prompts.get_image_scoring_prompt(subject)
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }],
                max_tokens=100,
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result.get('score', 0.0)

        except Exception as e:
            logger.warning(f"⚠️ Image scoring failed: {e}")
            return 0.0

    def _score_single_article(self, candidate, search_request, temp_dir: Path) -> float:
        """Score single article by downloading and extracting text."""

        try:
            # Download webpage
            downloader = FileDownloader(temp_dir)
            extractor = ContentExtractor(self.client)

            download_result = downloader.download_webpage(candidate, temp_dir)
            if download_result.status != "success":
                return 0.0

            # Extract actual text content
            html_content = download_result.filepath.read_text(encoding='utf-8')
            extracted_text = extractor.extract_article_text(html_content)

            if not extracted_text or len(extracted_text.strip()) < 100:
                return 0.1

            # Score based on extracted text using existing prompts
            candidate_descriptions = [f"{candidate.title} - {extracted_text[:300]}... ({candidate.url})"]
            prompt = prompts.get_candidate_scoring_prompt(candidate_descriptions, search_request)

            response = self.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )

            scores = json.loads(response.choices[0].message.content.strip())
            return scores[0] if isinstance(scores, list) and len(scores) > 0 else 0.0

        except Exception as e:
            logger.warning(f"⚠️ Article scoring failed: {e}")
            return 0.0
