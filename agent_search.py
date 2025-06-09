#!/usr/bin/env python3
"""
AI agent that autonomously searches for and downloads material from the internet.
Can handle prompts like:
- "Find and download 5 photographs of a zebra"
- "Download 2 random wikipedia articles"
- "Download the front page of https://news.ycombinator.com/"
"""

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


logger = setup_logging()


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
            "details": "any specific requirements"
        }

        Examples:
        - "Find X photos of Y" → {"action": "search_images", "subject": "Y",
          "count": X, "details": "photographs"}
        - "Download N articles" → {"action": "download_articles",
          "subject": "random wikipedia", "count": N, "details": "articles"}
        - "Download webpage URL" → {"action": "download_webpage",
          "subject": "URL", "count": 1, "details": "webpage"}

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

    def download_articles(self, count: int) -> list[dict]:
        """Download random Wikipedia articles."""
        logger.info(f"📚 Downloading {count} Wikipedia articles...")

        articles_dir = self.download_dir / "articles"
        articles_dir.mkdir(parents=True, exist_ok=True)

        all_downloads = []

        for i in range(count):
            try:
                # Get random Wikipedia article
                random_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
                response = self.session.get(random_url, timeout=10)
                response.raise_for_status()

                article_info = response.json()
                title = article_info['title']
                page_url = article_info['content_urls']['desktop']['page']

                logger.info(f"📄 Article {i+1}: {title}")

                # Download the article HTML
                article_response = self.session.get(page_url, timeout=15)
                article_response.raise_for_status()

                # Save as HTML file
                safe_title = re.sub(r'[^\w\-_.]', '_', title)
                filename = f"{safe_title}.html"
                filepath = articles_dir / filename
                filepath.write_text(article_response.text, encoding='utf-8')

                metadata = {
                    'filename': filename,
                    'filepath': str(filepath),
                    'url': page_url,
                    'title': title,
                    'size_bytes': len(article_response.text.encode('utf-8')),
                    'download_time': datetime.now().isoformat(),
                    'status': 'success'
                }

                all_downloads.append(metadata)
                logger.info(f"💾 {filename}")

            except Exception as e:
                logger.error(f"❌ Failed to download article {i+1}: {e}")
                all_downloads.append({
                    'error': str(e),
                    'status': 'failed',
                    'download_time': datetime.now().isoformat()
                })

        successful_count = len([d for d in all_downloads if d['status'] == 'success'])
        logger.info(f"✅ Downloaded {successful_count} articles → {articles_dir}")
        return all_downloads

    def download_webpage(self, url: str) -> list[dict]:
        """Download a specific webpage."""
        logger.info(f"🌐 Downloading webpage: {url}")

        webpages_dir = self.download_dir / "webpages"
        webpages_dir.mkdir(parents=True, exist_ok=True)

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # Generate filename from URL
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            filename = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = webpages_dir / filename

            filepath.write_text(response.text, encoding='utf-8')

            metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'url': url,
                'size_bytes': len(response.text.encode('utf-8')),
                'download_time': datetime.now().isoformat(),
                'status': 'success'
            }

            logger.info(f"✅ Downloaded webpage → {webpages_dir}")
            return [metadata]

        except Exception as e:
            logger.error(f"❌ Failed to download {url}: {e}")
            return [{
                'url': url,
                'error': str(e),
                'status': 'failed',
                'download_time': datetime.now().isoformat()
            }]

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
            logger.error(f"❌ Download failed {url}: {e}")
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
            return self.download_articles(request_info['count'])
        elif request_info['action'] == 'download_webpage':
            return self.download_webpage(request_info['subject'])
        else:
            logger.error(f"❌ Unknown action: {request_info['action']}")
            return []


def main():
    import sys
    from os import getenv

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
