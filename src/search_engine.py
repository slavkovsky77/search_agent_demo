"""
Search engine wrapper for SearXNG with candidate validation and scoring.
"""
import json
from urllib.parse import urlparse
import requests
from openai import OpenAI

from .models import SearchRequest, SearchCandidate, ImageSearchResult
from . import prompts
from .config import setup_logging

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

    def search_articles(self, query: str, is_news: bool = False,
                        max_candidates: int = 30) -> list[SearchCandidate]:
        """Search for articles/news and return candidates."""
        logger.info(
            f"🔍 Searching {'news' if is_news else 'articles'}: '{query}' "
            f"(up to {max_candidates} candidates)"
        )

        search_url = f"{self.searxng_url}/search"
        category = "news" if is_news else "general"
        engines = 'bing news,google news' if is_news else 'google,bing,duckduckgo'

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

    def score_and_select_candidates(self, candidates: list[SearchCandidate],
                                    search_request: SearchRequest,
                                    desired_count: int) -> list[SearchCandidate]:
        """Score candidates using AI and select the best ones."""
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

    def generate_search_queries(self, search_request: SearchRequest) -> list[str]:
        """Generate search queries using AI."""
        if search_request.content_type == "images":
            prompt = prompts.get_search_queries_prompt(search_request.subject, search_request.count)
        else:
            prompt = prompts.get_content_queries_prompt(
                search_request.topic, search_request.source,
                search_request.content_type, search_request.count
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
