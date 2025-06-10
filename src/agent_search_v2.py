"""
Clean, modular AI agent for autonomous internet search and downloads.
Separates search and download phases with proper validation.
"""
import sys
import json
from os import getenv
from pathlib import Path
from openai import OpenAI

from .models import SearchRequest, DownloadResult
from . import prompts
from .search_engine import SearchEngine
from .file_downloader import FileDownloader
from .content_extractor import ContentExtractor
from .config import setup_logging

logger = setup_logging(__name__)


class InternetSearchAgent:
    """
    Modular AI agent with separated responsibilities:
    - Search phase: find and score candidates
    - Download phase: download and validate files
    """

    def __init__(self, openrouter_api_key: str, searxng_url: str = "http://localhost:8080"):
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/rapid-sos-search",
                "X-Title": "Internet Search Agent v2"
            }
        )

        # Initialize components
        self.search_engine = SearchEngine(searxng_url, self.client)
        self.download_dir = Path("downloads")
        self.downloader = FileDownloader(self.download_dir)
        self.content_extractor = ContentExtractor(self.client)

        logger.info("🚀 InternetSearchAgent v2 ready")

    def execute_request(self, prompt: str) -> list[DownloadResult]:
        """Main entry point: understand request and execute it."""
        logger.info(f"📝 Processing request: {prompt}")

        # Phase 1: Understand the request
        search_request = self._understand_request(prompt)
        if not search_request:
            return []

        # Phase 2: Search for candidates
        candidates = self._search_phase(search_request)
        if not candidates:
            logger.warning("⚠️ No candidates found in search phase")
            return []

        # Phase 3: Download and validate
        results = self._download_phase(candidates, search_request)

        # Save results log
        self._save_results_log(results)
        return results

    def _understand_request(self, prompt: str) -> SearchRequest | None:
        """Parse user request using AI."""
        system_prompt = prompts.get_request_understanding_prompt()

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
            request_data = json.loads(content)

            search_request = SearchRequest(**request_data)
            logger.info(f"🧠 Understood: {search_request.action} → {search_request.subject} (×{search_request.count})")
            return search_request

        except Exception as e:
            logger.exception(f"❌ Failed to understand request: {e}")
            return None

    def _search_phase(self, search_request: SearchRequest) -> list:
        """Search for candidates based on request type."""
        logger.info(f"🔍 Search phase: Finding candidates for {search_request.content_type}")

        # Generate search queries
        queries = self.search_engine.generate_search_queries(search_request)
        all_candidates = []

        for query in queries:
            if search_request.content_type == "images":
                candidates = self.search_engine.search_images(query, max_candidates=20)
            else:
                is_news = search_request.content_type == "news"
                candidates = self.search_engine.search_articles(query, is_news=is_news, max_candidates=30)

            all_candidates.extend(candidates)

        # Score and select best candidates for articles (images don't need scoring yet)
        if search_request.content_type != "images" and len(all_candidates) > search_request.count:
            all_candidates = self.search_engine.score_and_select_candidates(
                all_candidates, search_request, search_request.count
            )

        logger.info(f"🎯 Search phase complete: {len(all_candidates)} candidates selected")
        return all_candidates[:search_request.count]

    def _download_phase(self, candidates: list, search_request: SearchRequest) -> list[DownloadResult]:
        """Download and validate all candidates."""
        logger.info(f"⬇️ Download phase: Processing {len(candidates)} candidates")

        # Create appropriate directory
        if search_request.content_type == "images":
            directory = self.download_dir / "images" / search_request.subject.replace(" ", "_")
        else:
            topic_source = f"{search_request.source}_{search_request.topic}".replace(" ", "_")
            directory = self.download_dir / "articles" / topic_source

        directory.mkdir(parents=True, exist_ok=True)

        results = []
        successful_downloads = 0

        for i, candidate in enumerate(candidates):
            if successful_downloads >= search_request.count:
                break

            logger.info(f"📥 Downloading {i+1}/{len(candidates)}")

            # Download based on content type
            if search_request.content_type == "images":
                result = self.downloader.download_image(candidate, directory)
            else:
                result = self.downloader.download_webpage(candidate, directory)

                # Extract text content for articles
                if result.status == "success" and result.filepath.exists():
                    html_content = result.filepath.read_text(encoding='utf-8')
                    extracted_text = self.content_extractor.extract_article_text(html_content)
                    result.extracted_text = extracted_text

            results.append(result)

            if result.status == "success":
                successful_downloads += 1

        logger.info(f"✅ Download phase complete: {successful_downloads} successful downloads → {directory}")
        return results

    def _save_results_log(self, results: list[DownloadResult]) -> None:
        """Save results to JSON log file."""
        try:
            log_file = self.download_dir / 'download_log.json'

            # Convert results to dict for JSON serialization
            results_data = []
            for result in results:
                result_dict = result.model_dump()
                # Convert Path to string for JSON
                result_dict['filepath'] = str(result_dict['filepath'])
                results_data.append(result_dict)

            log_file.write_text(json.dumps(results_data, indent=2, default=str))
            logger.info(f"📝 Results saved to {log_file}")

        except Exception as e:
            logger.exception(f"❌ Failed to save results log: {e}")


def main():
    """Command line entry point."""
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

        # Show summary
        successful = len([r for r in results if r.status == "success"])
        logger.info(f"🎉 Complete! {successful}/{len(results)} downloads successful")

    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
