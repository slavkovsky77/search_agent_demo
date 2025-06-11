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

from .models import SearchRequest, SearchCandidate, ImageSearchResult, DownloadResult
from . import prompts
from .config import setup_logging, LLMModels, SystemConstants
from .content_extractor import ContentExtractor

logger = setup_logging(__name__)


class SearchEngine:
    """Handles search operations via SearXNG with intelligent candidate selection."""

    def __init__(self, searxng_url: str, openai_client: OpenAI):
        self.searxng_url = searxng_url.rstrip('/')
        self.client = openai_client
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': SystemConstants.DEFAULT_USER_AGENT
        })

    def search_images(self, query: str, max_candidates: int = 20) -> list[ImageSearchResult]:
        """Search for image candidates using SearXNG."""
        logger.info(f"🖼️ Searching images: '{query}' (max {max_candidates})")

        # Use image-specific engines
        search_url = f"{self.searxng_url}/search"
        params = {
            'q': query,
            'format': 'json',
            'engines': 'bing_images,google_images',
            'categories': 'images'
        }

        try:
            response = self.session.get(search_url, params=params, timeout=SystemConstants.SEARCH_REQUEST_TIMEOUT)
            response.raise_for_status()
            search_results = response.json()

            candidates = []
            for result in search_results.get('results', [])[:max_candidates]:
                try:
                    # Skip results without proper image URLs
                    if not self._is_valid_image_url(result.get('url', '')):
                        continue

                    candidates.append(ImageSearchResult(
                        url=result['img_src'],
                        title=result.get('title'),
                        thumbnail_url=result.get('thumbnail_src'),
                        source_site=urlparse(result['img_src']).netloc)
                    )

                except Exception as e:
                    logger.debug(f"⚠️ Skipped invalid image result: {e}")
                    continue

            logger.info(f"✅ Found {len(candidates)} image candidates")
            return candidates

        except Exception as e:
            logger.exception(f"❌ Image search failed for '{query}': {e}")
            return []

    def search_articles(self, query: str, max_candidates: int = 30) -> list[SearchCandidate]:
        """Search for articles and return candidates from both news and general categories."""
        logger.info(
            f"🔍 Searching articles: '{query}' "
            f"(up to {max_candidates} candidates)"
        )

        all_candidates = []
        site_constraint = self._extract_site_constraint(query)

        # Search news category first
        news_candidates = self._search_category(query, 'news', 'bing news,google news', "📰")
        all_candidates.extend(news_candidates)
        logger.info(f"📰 Found {len(news_candidates)} news candidates")

        # Search general category
        if len(all_candidates) < max_candidates:
            remaining_needed = max_candidates - len(all_candidates)
            existing_urls = {candidate.url for candidate in all_candidates}

            general_candidates = self._search_category(
                query, 'general', 'google,bing,duckduckgo', "🎯",
                exclude_urls=existing_urls, max_results=remaining_needed
            )
            all_candidates.extend(general_candidates)
            logger.info(f"🎯 Found {len(general_candidates)} additional general candidates")

        # Filter by site constraint if present
        if site_constraint:
            all_candidates = self._filter_by_site_constraint(all_candidates, site_constraint)

        # Limit to max_candidates
        final_candidates = all_candidates[:max_candidates]
        logger.info(f"✅ Total found {len(final_candidates)} article candidates")
        return final_candidates

    def _extract_site_constraint(self, query: str) -> str | None:
        """Extract site constraint from search query (e.g., 'site:wikipedia.org' -> 'wikipedia.org')."""
        if 'site:' not in query.lower():
            return None

        import re
        match = re.search(r'site:([^\s]+)', query.lower())
        if match:
            site_constraint = match.group(1)
            logger.info(f"🎯 Site constraint detected: {site_constraint}")
            return site_constraint
        return None

    def _filter_by_site_constraint(self, candidates: list[SearchCandidate],
                                   site_constraint: str) -> list[SearchCandidate]:
        """Filter candidates to only include URLs matching the site constraint."""
        filtered_candidates = [
            candidate for candidate in candidates
            if site_constraint in str(candidate.url).lower()
        ]
        logger.info(f"🔍 Filtered to {len(filtered_candidates)} candidates matching {site_constraint}")
        return filtered_candidates

    def _search_category(self, query: str, category: str, engines: str, emoji: str,
                         exclude_urls: set = None, max_results: int = None) -> list[SearchCandidate]:
        """Helper method to search a specific category and return candidates."""
        if exclude_urls is None:
            exclude_urls = set()

        search_url = f"{self.searxng_url}/search"
        params = {
            'q': query,
            'categories': category,
            'format': 'json',
            'engines': engines
        }

        try:
            if max_results:
                logger.info(f"{emoji} Searching {category} category for {query} (need {max_results} more)")
            else:
                logger.info(f"{emoji} Searching {category} category for {query}")

            response = self.session.get(search_url, params=params, timeout=SystemConstants.SEARCH_REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            candidates = []
            for result in data.get('results', []):
                if result.get('url') and result['url'] not in exclude_urls:
                    candidates.append(SearchCandidate(
                        url=result['url'],
                        title=result.get('title', 'Unknown Title'),
                        description=result.get('content', ''),
                        engine=result.get('engine', 'unknown')
                    ))
                    if max_results and len(candidates) >= max_results:
                        break

            return candidates

        except Exception as e:
            logger.error(f"❌ Article search failed: {e}")
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
                model=LLMModels.QUERY_GENERATION,
                messages=[{"role": "user", "content": prompt}],
                **LLMModels.QUERY_PARAMS
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

    def score_and_select_candidates(self, candidates: list[SearchCandidate | ImageSearchResult],
                                    search_request: SearchRequest,
                                    desired_count: int) -> list[SearchCandidate]:
        """Score candidates using AI and return top candidates."""
        logger.info(f"🧠 Scoring {len(candidates)} candidates to select top {desired_count}")

        if not candidates:
            return []

        scored_candidates = []
        batch_size = SystemConstants.SCORING_BATCH_SIZE  # Process candidates in batches

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(candidates) + batch_size - 1) // batch_size

            logger.info(f"🔍 Scoring batch {batch_num}/{total_batches} ({len(batch)} candidates)")

            # Prepare candidate descriptions for AI scoring
            candidate_descriptions = [
                f"{c.title} - {c.description[:100]}... ({c.url})"
                for c in batch
            ]

            try:
                prompt = prompts.get_article_scoring_prompt(candidate_descriptions, search_request)

                response = self.client.chat.completions.create(
                    model=LLMModels.CONTENT_SCORING,
                    messages=[{"role": "user", "content": prompt}],
                    **LLMModels.SCORING_PARAMS
                )

                response_content = response.choices[0].message.content.strip()
                logger.debug(f"🤖 Raw scoring response: {response_content[:200]}...")

                scores = json.loads(response_content)

                # Assign scores to candidates in this batch
                for j, candidate in enumerate(batch):
                    if j < len(scores):
                        candidate.score = scores[j]
                    else:
                        candidate.score = 0.0

                scored_candidates.extend(batch)

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing failed for batch {batch_num}: {e}")
                logger.error(f"🔧 Raw response was: {response.choices[0].message.content}")
                # Assign default scores to this batch
                for candidate in batch:
                    candidate.score = 0.5  # Neutral score
                scored_candidates.extend(batch)

            except Exception as e:
                logger.exception(f"❌ Scoring failed for batch {batch_num}: {e}")
                # Assign low scores to failed batch
                for candidate in batch:
                    candidate.score = 0.0
                scored_candidates.extend(batch)

        # Sort by score and return top candidates
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        selected = scored_candidates[:desired_count]

        logger.info(f"✅ Selected top {len(selected)} candidates (scores: {[f'{c.score:.2f}' for c in selected[:5]]})")
        return selected

    def _score_single_image(self, download_result: DownloadResult, subject: str) -> float:
        """Score single image with vision API (resized to max 480p)."""
        logger.debug(f"🔍 Scoring image: {download_result.title}")

        try:
            # Download image
            response = requests.get(str(download_result.url), timeout=SystemConstants.SEARCH_REQUEST_TIMEOUT, headers={
                'User-Agent': SystemConstants.DEFAULT_USER_AGENT
            })
            response.raise_for_status()

            # Resize to max 480p to save API costs
            image = Image.open(BytesIO(response.content))
            if max(image.size) > SystemConstants.MAX_IMAGE_DIMENSION:
                ratio = SystemConstants.MAX_IMAGE_DIMENSION / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to bytes
            img_buffer = BytesIO()
            image.save(img_buffer, format='JPEG', quality=SystemConstants.IMAGE_QUALITY)
            image_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            # Score with vision using prompts
            vision_prompt = prompts.get_image_scoring_prompt(subject)
            response = self.client.chat.completions.create(
                model=LLMModels.IMAGE_SCORING,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }],
                **LLMModels.IMAGE_SCORING_PARAMS
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result.get('score', 0.0)

        except Exception as e:
            logger.warning(f"⚠️ Image scoring failed: {e}")
            return 0.0

    def _score_single_article(self, download_result: DownloadResult, search_request: SearchRequest) -> float:
        """Score single article by downloading and extracting text."""
        try:
            # Download webpage
            extractor = ContentExtractor(self.client)

            # Extract actual text content
            html_content = download_result.filepath.read_text(encoding='utf-8')
            extracted_text = extractor.extract_article_text(html_content)

            if not extracted_text or len(extracted_text.strip()) < SystemConstants.MIN_ARTICLE_CONTENT_LENGTH:
                logger.warning(f"❌ Poor content extraction from {download_result.title[:50]}... (JS-heavy site?)")
                return 0.0

            # Save extracted text if we have a target filepath
            extractor.save_extracted_text(extracted_text, download_result.filepath)

            # Score based on extracted text using existing prompts
            candidate_descriptions = [f"{download_result.title} - {extracted_text[:300]}... ({download_result.url})"]
            prompt = prompts.get_article_scoring_prompt(candidate_descriptions, search_request)

            response = self.client.chat.completions.create(
                model=LLMModels.CONTENT_SCORING,
                messages=[{"role": "user", "content": prompt}],
                **LLMModels.SCORING_PARAMS
            )

            scores = json.loads(response.choices[0].message.content.strip())
            return scores[0] if isinstance(scores, list) and len(scores) > 0 else 0.0

        except Exception as e:
            logger.warning(f"⚠️ Article scoring failed: {e}")
            return 0.0
