"""
AI agent that autonomously searches for and downloads material from the internet.
Can handle prompts like:
- "Find and download 5 photographs of a zebra"
- "Download 2 random wikipedia articles"
- "Download the front page of https://news.ycombinator.com/"
"""
import sys
from os import getenv
import json
import time
import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse, unquote
from datetime import datetime
import requests
from openai import OpenAI
from config import setup_logging


logger = setup_logging(__name__)


class InternetSearchAgent:
    def __init__(self, openrouter_api_key: str, searxng_url: str = "http://localhost:8080"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/rapid-sos-search",
                "X-Title": "Internet Search Agent"
            }
        )
        self.searxng_url = searxng_url.rstrip('/')
        self.download_dir = Path("downloads")
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        })
        logger.info(f"🚀 InternetSearchAgent ready → {self.searxng_url}")

    def understand_request(self, prompt: str) -> dict:
        """Use AI to understand what the user wants to do."""

        system_prompt = """You are an expert at understanding user requests for downloading content from the internet.

        Analyze the user's request and return EXACTLY this JSON format:
        {
            "action": "search_images" | "download_articles" | "download_webpage",
            "subject": "what to search for or download",
            "count": number_of_items,
            "content_type": "wikipedia" | "news" | "webpage" | "images",
            "source": "specific website or 'any'",
            "topic": "specific topic or 'random'"
        }

        Examples:
        - "Find photos" → {"action": "search_images", "subject": "Y", "count": X,
          "content_type": "images", "source": "any", "topic": "Y"}
        - "Download random articles" → {"action": "download_articles", "subject": "articles",
          "count": N, "content_type": "articles", "source": "any", "topic": "random"}
        - "Download wikipedia articles" → {"action": "download_articles", "subject": "articles",
          "count": N, "content_type": "articles", "source": "wikipedia.org", "topic": "random"}
        - "Download articles about TOPIC" → {"action": "download_articles", "subject": "articles",
          "count": N, "content_type": "articles", "source": "any", "topic": "TOPIC"}
        - "Download articles about TOPIC from SITE" → {"action": "download_articles", "subject": "articles",
          "count": N, "content_type": "articles", "source": "SITE", "topic": "TOPIC"}
        - "Download from SITE" → {"action": "download_articles", "subject": "articles",
          "count": 1, "content_type": "news", "source": "SITE", "topic": "latest"}
        - "Download webpage" → {"action": "download_webpage", "subject": "URL", "count": 1,
          "content_type": "webpage", "source": "URL", "topic": "homepage"}

        IMPORTANT: Extract the source website correctly!
        - "wikipedia articles" → source: "wikipedia.org"
        - "articles from BBC" → source: "bbc.com"
        - "CNN news" → source: "cnn.com"
        - If no specific source mentioned → source: "any"

        Return ONLY the JSON, no other text."""

        try:
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()
            logger.debug(f"AI understanding: {content}")

            # Parse the JSON response
            request_info = json.loads(content)
            action = request_info['action']
            subject = request_info['subject']
            count = request_info['count']
            logger.info(f"🧠 Understood: {action} → {subject} (×{count})")
            return request_info

        except Exception as e:
            logger.exception(f"❌ Failed to understand request: {e}")

    def search_images(self, subject: str, count: int) -> list[dict]:
        """Search for and download images."""
        logger.info(f"🔍 Searching for {count} images of: {subject}")

        # Create subject-specific directory
        subject_dir = self.download_dir / "images" / subject.replace(" ", "_")
        subject_dir.mkdir(parents=True, exist_ok=True)

        # Generate search queries
        queries = self._generate_search_queries(subject, count)

        all_downloads = []
        downloaded_count = 0

        for i, query in enumerate(queries):
            if downloaded_count >= count:
                break

            logger.info(f"🔍 Query {i+1}: '{query}'")
            image_urls = self._search_images_searxng(query, 5)

            if not image_urls:
                logger.warning(f"⚠️ No images for: '{query}'")
                continue

            for url in image_urls:
                if downloaded_count >= count:
                    break

                metadata = self._download_file(url, subject_dir)
                all_downloads.append(metadata)

                if metadata['status'] == 'success':
                    downloaded_count += 1

                time.sleep(1)  # Be respectful

        logger.info(f"✅ Downloaded {downloaded_count} images → {subject_dir}")
        return all_downloads

    def download_content(self, count: int, topic: str = "random",
                         source: str = "any", content_type: str = "articles",
                         direct_url: str = None) -> list[dict]:
        """Intelligently download content using search engine or direct URL."""

        if direct_url:
            # Direct URL download (for webpages)
            logger.info(f"🌐 Downloading webpage: {direct_url}")
            webpages_dir = self.download_dir / "webpages"
            webpages_dir.mkdir(parents=True, exist_ok=True)

            title = f"Webpage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            metadata = self._download_single_webpage(direct_url, title, webpages_dir)
            logger.info(f"✅ Downloaded webpage → {webpages_dir}")
            return [metadata]

        # Search-based download (for articles/news)
        logger.info(f"📚 Downloading {count} {content_type} about '{topic}' from {source}...")

        # Create directory
        articles_dir = self.download_dir / "articles" / f"{source}_{topic}".replace(" ", "_")
        articles_dir.mkdir(parents=True, exist_ok=True)

        # Construct smart search query using AI
        search_queries = self._generate_content_queries(topic, source, content_type, count)
        logger.debug(f"🔍 Generated queries: {search_queries}")

        # Use SearXNG to find articles
        all_downloads = []

        for i, search_query in enumerate(search_queries):
            if len(all_downloads) >= count:
                break

            logger.debug(f"🔍 Search query {i+1}: {search_query}")

            try:
                search_url = f"{self.searxng_url}/search"

                # Choose appropriate SearXNG category
                category = "news" if content_type == "news" else "general"

                params = {
                    'q': search_query,
                    'categories': category,
                    'format': 'json',
                    'engines': 'google,bing,duckduckgo' if category == "general" else 'bing news,google news'
                }

                response = self.session.get(search_url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()
                results_count = len(data.get('results', []))
                logger.debug(f"🔍 SearXNG returned {results_count} results for query {i+1}")

                # Debug: show what we actually got
                for j, result in enumerate(data.get('results', [])[:3]):  # Show first 3
                    title_debug = result.get('title', 'No title')
                    url_debug = result.get('url', 'No URL')
                    logger.debug(f"  Result {j+1}: {title_debug} → {url_debug}")

                for result in data.get('results', []):
                    if len(all_downloads) >= count:
                        break

                    title = result.get('title', f'Article_{len(all_downloads)+1}')
                    url = result.get('url')

                    if url:
                        logger.info(f"📄 Article {len(all_downloads)+1}: {title},   url: {url}")
                        metadata = self._download_single_webpage(url, title, articles_dir)
                        all_downloads.append(metadata)
                        time.sleep(1)  # Be respectful

            except Exception as e:
                logger.error(f"❌ Failed search query {i+1}: {e}")

        successful_count = len([d for d in all_downloads if d.get('status') == 'success'])
        logger.info(f"✅ Downloaded {successful_count} articles → {articles_dir}")
        return all_downloads

    def _generate_content_queries(self, topic: str, source: str, content_type: str, count: int) -> list[str]:
        """Use AI to generate smart search queries for content."""
        try:
            if topic == "random":
                # Don't search for "random" - generate actual diverse topics
                system_prompt = f"""Generate {min(count, 3)} diverse topics to find interesting {content_type}.

                Generate completely different subjects that would have good {content_type}.
                DO NOT use the word "random" in any query.

                Examples of good topics: history, science, technology, nature, culture, biography

                If source is specified, add "site:{source}" to each query.
                Source: {source if source != "any" else "not specified"}

                Return EXACTLY a JSON array like: ["topic1", "topic2", .., "topic{count}"]
                Return ONLY the JSON array."""
            else:
                # Specific topic search
                system_prompt = f"""Generate {min(count, 3)} search queries to find {content_type} about: {topic}

                If source is specified, add "site:{source}" to each query.
                Source: {source if source != "any" else "not specified"}

                Return EXACTLY a JSON array like: ["query1", "query2", .., "query{count}"]
                Return ONLY the JSON array."""

            response = self.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=200,
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()
            queries = json.loads(content)

            if isinstance(queries, list):
                return queries[:count]

        except Exception as e:
            logger.exception(f"❌ Content query generation failed: {e}")

    def _download_single_webpage(self, url: str, title: str, directory: Path) -> dict:
        """Download a single webpage and return metadata."""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # Clean filename
            safe_title = re.sub(r'[^\w\-_.]', '_', title)
            filename = f"{safe_title}.html"
            filepath = directory / filename

            filepath.write_text(response.text, encoding='utf-8')

            metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'url': url,
                'title': title,
                'size_bytes': len(response.text.encode('utf-8')),
                'download_time': datetime.now().isoformat(),
                'status': 'success'
            }

            logger.info(f"💾 {filename}")
            return metadata

        except Exception as e:
            logger.error(f"❌ Failed to download {url}: {e}")
            return {
                'url': url,
                'title': title,
                'error': str(e),
                'status': 'failed',
                'download_time': datetime.now().isoformat()
            }

    def _generate_search_queries(self, subject: str, count: int) -> list[str]:
        """Generate search queries for the subject."""
        try:
            system_prompt = f"""Generate {min(count, 5)} different search queries to find of: {subject}

            Return EXACTLY a JSON array like: ["query1", "query2",.. ,"query{count}"]
            Each query should be 1-4 words, designed to find {subject}.
            Return ONLY the JSON array."""

            logger.debug(f"🔍 Generating search queries with prompt: {system_prompt}")
            response = self.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=200,
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()
            queries = json.loads(content)
            logger.debug(f"🔍 Generated queries: {queries}")
            if isinstance(queries, list):
                return queries[:count]

        except Exception as e:
            logger.error(f"❌ Query generation failed: {e}")

        # Fallback queries
        return [f"{subject} photograph", f"{subject} image", f"{subject} photo"]

    def _search_images_searxng(self, query: str, max_results: int = 5) -> list[str]:
        """Search for images using SearXNG."""
        try:
            search_url = f"{self.searxng_url}/search"
            params = {
                'q': f"{query} photograph",
                'categories': 'images',
                'format': 'json',
                'engines': 'bing images,google images,duckduckgo images'
            }

            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            images = []

            for result in data.get('results', []):
                if 'img_src' in result:
                    img_url = result['img_src']
                    if self._is_valid_image_url(img_url):
                        images.append(img_url)
            return images[:max_results]

        except Exception as e:
            logger.error(f"❌ SearXNG search failed: {e}")
            return []

    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL looks like a valid image URL."""
        if not url or not url.startswith('http'):
            return False

        parsed = urlparse(url)
        path = parsed.path.lower()

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        has_image_ext = any(path.endswith(ext) for ext in image_extensions)

        has_image_keywords = any(keyword in url.lower()
                                 for keyword in ['image', 'photo', 'picture', 'img'])

        return has_image_ext or has_image_keywords

    def _download_file(self, url: str, directory: Path) -> dict:
        """Download a file and return metadata."""
        try:
            headers = {
                'User-Agent': self.session.headers['User-Agent'],
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Referer': 'https://www.google.com/'
            }

            response = requests.get(url, timeout=30, headers=headers, stream=True)
            response.raise_for_status()

            content = response.content
            if len(content) < 1000:
                raise Exception(f"Content too small: {len(content)} bytes")

            # Generate filename
            parsed_url = urlparse(url)
            filename = Path(unquote(parsed_url.path)).name
            if not filename or '.' not in filename:
                content_type = response.headers.get('content-type', 'unknown')
                ext = mimetypes.guess_extension(content_type) or '.jpg'
                filename = f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"

            # Save file
            filepath = directory / filename
            filepath.write_bytes(content)

            metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'url': url,
                'size_bytes': len(content),
                'download_time': datetime.now().isoformat(),
                'status': 'success'
            }

            logger.info(f"💾 {filename} ({len(content)} bytes)")
            return metadata

        except Exception as e:
            logger.exception(f"❌ Download failed {url}: {e}")
            return {
                'url': url,
                'error': str(e),
                'status': 'failed',
                'download_time': datetime.now().isoformat()
            }

    def execute_request(self, prompt: str) -> list[dict]:
        """Main method: understand and execute user request."""
        logger.info(f"📝 Request: {prompt}")

        # Understand what the user wants
        request_info = self.understand_request(prompt)

        # Execute the appropriate action
        if request_info['action'] == 'search_images':
            return self.search_images(request_info['subject'], request_info['count'])
        elif request_info['action'] == 'download_articles':
            return self.download_content(
                count=request_info['count'],
                topic=request_info.get('topic', 'random'),
                source=request_info.get('source', 'any'),
                content_type=request_info.get('content_type', 'articles')
            )
        elif request_info['action'] == 'download_webpage':
            # Webpage download is just single-item content download
            return self.download_content(
                count=1,
                topic=request_info.get('topic', 'homepage'),
                source=request_info.get('source', 'any'),
                content_type='webpage',
                direct_url=request_info['subject']
            )
        else:
            logger.error(f"❌ Unknown action: {request_info['action']}")
            return []


def main():

    api_key = getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("❌ OPENROUTER_API_KEY not set. Get one from https://openrouter.ai/keys")
        return

    # Get user prompt
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
    else:
        prompt = input("📝 What would you like me to download? ")

    # Create agent and execute request
    try:
        searxng_url = getenv('SEARXNG_URL', 'http://localhost:8080')
        agent = InternetSearchAgent(api_key, searxng_url)
        results = agent.execute_request(prompt)

        # Save results log
        log_file = agent.download_dir / 'download_log.json'
        log_file.write_text(json.dumps(results, indent=2))

    except Exception as e:
        logger.exception(f"💥 Error: {e}")
        raise


if __name__ == "__main__":
    main()
