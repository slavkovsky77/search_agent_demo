"""
Clean, modular AI agent for autonomous internet search and downloads.
New workflow: download candidates -> score each -> select best.
"""
import argparse
import json
from os import getenv
from pathlib import Path
from openai import OpenAI
import shutil

from .models import (
    SearchRequest,
    DownloadResult,
    SearchCandidate,
    ImageSearchResult,
    SearchAction,
    ContentType)
from . import prompts
from .search_engine import SearchEngine
from .file_downloader import FileDownloader
from .content_extractor import ContentExtractor
from .config import setup_logging, LLMModels, SystemConstants

logger = setup_logging(__name__)


class InternetSearchAgent:
    """
    Modular AI agent with separated responsibilities:
    - Search phase: find and score candidates
    - Download phase: download and validate files
    """

    def __init__(self, openrouter_api_key: str, searxng_url: str = SystemConstants.DEFAULT_SEARXNG_URL):
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

        # Phase 2: Search for candidates (or direct URL)
        if search_request.action == SearchAction.DOWNLOAD_WEBPAGE and search_request.subject.startswith("http"):
            # Direct URL download - create candidate from URL
            candidates = [SearchCandidate(
                url=search_request.subject,
                title=f"Direct download: {search_request.subject}",
                description="Direct URL download"
            )]
        else:
            # Normal search flow
            candidates = self._search_phase(search_request)
            if not candidates:
                logger.warning("⚠️ No candidates found in search phase")
                return []

        # Phase 3: Download and validate
        results = self._download_phase(candidates, search_request)

        # Save results log and search candidates
        self._save_results_log(results)
        self._save_search_candidates(candidates, search_request)
        return results

    def _understand_request(self, prompt: str) -> SearchRequest | None:
        """Parse user request using AI."""
        system_prompt = prompts.get_request_understanding_prompt()

        try:
            response = self.client.chat.completions.create(
                model=LLMModels.REQUEST_UNDERSTANDING,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                **LLMModels.REQUEST_PARAMS
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
        """Search for candidates (no scoring - that happens in download phase)."""
        logger.info(f"🔍 Search phase: Finding candidates for {search_request.content_type}")

        # Generate search queries
        queries = self.search_engine.generate_search_queries(search_request)
        all_candidates = []

        for query in queries:
            if search_request.content_type == "images":
                candidates = self.search_engine.search_images(
                    query, max_candidates=SystemConstants.MAX_IMAGE_CANDIDATES)
            else:
                # All non-image content types are treated as articles
                candidates = self.search_engine.search_articles(
                    query, max_candidates=SystemConstants.MAX_ARTICLE_CANDIDATES)

            all_candidates.extend(candidates)

        logger.info(f"🎯 Search phase complete: {len(all_candidates)} candidates found")
        return all_candidates

    def _download_phase(self, candidates: list, search_request: SearchRequest) -> list[DownloadResult]:
        """Download and score candidates, return best results."""
        logger.info(f"⬇️ Download phase: {len(candidates)} candidates → {search_request.count} results")

        directory = self._create_download_directory(search_request)
        candidates_to_download = self._select_candidates_for_download(candidates, search_request)
        download_results = self._download_candidates(candidates_to_download, directory, search_request)
        selected_results = self._select_best_results(download_results, search_request.count, directory)

        logger.info(f"✅ Download complete: {len(selected_results)} results → {directory}")
        return selected_results

    def _create_download_directory(self, search_request: SearchRequest) -> Path:
        """Create appropriate download directory."""
        if search_request.content_type == ContentType.IMAGES:
            directory = self.download_dir / "images" / search_request.subject.replace(" ", "_")
        else:
            topic_source = f"{search_request.source}_{search_request.topic}".replace(" ", "_")
            directory = self.download_dir / "articles" / topic_source

        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _select_candidates_for_download(self, candidates: list,
                                        search_request: SearchRequest) -> list:
        """Select which candidates to download using smart diminishing returns."""
        count = search_request.count

        # Smart multiplier calculation with conservative diminishing returns
        # 1→5x, 2→3x, 5→2x, 20→1.5x, 100→1.2x
        smart_multiplier = 1 + (SystemConstants.SMART_MULTIPLIER_FACTOR / (count + 1))

        desired_count = min(int(count * smart_multiplier), len(candidates))

        logger.info(f"🧮 Smart selection: {count} requested → checking {desired_count} "
                    f"candidates ({smart_multiplier:.1f}x)")

        if search_request.content_type != ContentType.IMAGES:
            return self.search_engine.score_and_select_candidates(
                candidates, search_request, desired_count)
        else:
            return candidates[:desired_count]

    def _download_candidates(
            self,
            candidates: list[SearchCandidate | ImageSearchResult],
            directory: Path,
            search_request: SearchRequest) -> list[DownloadResult]:
        """Download all candidates and extract content."""
        results = []
        # Download all candidates into candidates/ subfolder first
        candidates_dir = directory / "candidates"
        candidates_dir.mkdir(exist_ok=True)

        for i, candidate in enumerate(candidates):
            logger.info(f"📥 {i+1}/{len(candidates)}: {candidate.title[:50]}...")

            # Download into candidates/ folder
            if search_request.content_type == ContentType.IMAGES:
                result = self.downloader.download_image(candidate, candidates_dir)
                if result:
                    result.relevance_score = self.search_engine._score_single_image(result, search_request.subject)
            else:
                result = self.downloader.download_webpage(candidate, candidates_dir)
                if result:
                    relevance_score = self.search_engine._score_single_article(result, search_request)
                    result.search_description = candidate.description
                    result.title = candidate.title
                    result.relevance_score = relevance_score

            if not result:
                logger.warning(f"❌ Failed to download and score {candidate.title[:50]}...")
                continue

            results.append(result)

        return results

    def _select_best_results(self, results: list[DownloadResult], count: int, directory: Path) -> list[DownloadResult]:
        """Select best results and move them to main folder."""
        results.sort(key=lambda x: x.relevance_score or 0.0, reverse=True)
        selected = results[:count]

        # Move selected files from candidates/ to main folder
        for result in selected:
            try:
                if result.filepath.exists():
                    # Move HTML to main folder
                    new_html_path = directory / result.filepath.name
                    shutil.move(str(result.filepath), str(new_html_path))

                    # Move .txt if exists (calculate path before updating result.filepath)
                    old_txt_path = result.filepath.with_suffix('.txt')
                    if old_txt_path.exists():
                        new_txt_path = directory / old_txt_path.name
                        shutil.move(str(old_txt_path), str(new_txt_path))

                    result.filepath = new_html_path  # Update filepath after moving

            except Exception as e:
                logger.warning(f"⚠️ Failed to move selected file: {e}")

        return selected

    def _save_results_log(self, results: list[DownloadResult]) -> None:
        """Save results to JSON log file."""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.download_dir / f'download_log_{timestamp}.json'

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

    def _save_search_candidates(
            self,
            candidates: list[SearchCandidate | ImageSearchResult],
            search_request: SearchRequest) -> None:
        """Save all search candidates with their scores for analysis."""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidates_file = self.download_dir / f'search_candidates_{timestamp}.json'

            # All our candidates are Pydantic models - just dump them
            candidates_data = {
                "search_request": search_request.model_dump(),
                "total_candidates": len(candidates),
                "candidates": [candidate.model_dump() for candidate in candidates]
            }

            candidates_file.write_text(json.dumps(candidates_data, indent=2, default=str))
            logger.info(f"🔍 Search candidates saved to {candidates_file}")

        except Exception as e:
            logger.exception(f"❌ Failed to save search candidates: {e}")


def main():
    """Command line entry point."""
    parser = argparse.ArgumentParser(description='AI Internet Search Agent')
    parser.add_argument('prompt', nargs='*', help='Search prompt (e.g. "find 5 zebra photos")')
    parser.add_argument('--searxng-url', default=SystemConstants.DEFAULT_SEARXNG_URL,
                        help='SearXNG instance URL')
    args = parser.parse_args()

    api_key = getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("❌ OPENROUTER_API_KEY not set. Get one from https://openrouter.ai/keys")
        return

    # Create agent and execute request
    try:
        agent = InternetSearchAgent(api_key, args.searxng_url)
        results = agent.execute_request(' '.join(args.prompt))

        # Show summary
        successful = len([r for r in results if r.status == "success"])
        logger.info(f"🎉 Complete! {successful}/{len(results)} downloads successful")

    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
