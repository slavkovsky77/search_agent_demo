"""
Clean, modular AI agent for autonomous internet search and downloads.
New workflow: download candidates -> score each -> select best.
"""
import argparse
import json
from os import getenv
from pathlib import Path
import tempfile
from openai import OpenAI

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

        # Phase 2: Search for candidates (or direct URL)
        if search_request.action == SearchAction.DOWNLOAD_WEBPAGE and search_request.subject.startswith("http"):
            # Direct URL download - create candidate from URL
            from .models import SearchCandidate
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
        """Search for candidates (no scoring - that happens in download phase)."""
        logger.info(f"🔍 Search phase: Finding candidates for {search_request.content_type}")

        # Generate search queries
        queries = self.search_engine.generate_search_queries(search_request)
        all_candidates = []

        for query in queries:
            if search_request.content_type == "images":
                candidates = self.search_engine.search_images(query, max_candidates=20)
            else:
                # All non-image content types are treated as articles
                candidates = self.search_engine.search_articles(query, max_candidates=30)

            all_candidates.extend(candidates)

        logger.info(f"🎯 Search phase complete: {len(all_candidates)} candidates found")
        return all_candidates

    def _download_phase(
            self,
            candidates: list[SearchCandidate | ImageSearchResult],
            search_request: SearchRequest) -> list[DownloadResult]:
        """Use batch scoring for articles, then download only the best candidates."""

        logger.info(
            f"⬇️ Download phase: Processing {len(candidates)} candidates "
            f"to find {search_request.count} best results"
        )

        # Create appropriate directory
        if search_request.content_type == ContentType.IMAGES:
            directory = self.download_dir / "images" / search_request.subject.replace(" ", "_")
        else:
            topic_source = f"{search_request.source}_{search_request.topic}".replace(" ", "_")
            directory = self.download_dir / "articles" / topic_source

        directory.mkdir(parents=True, exist_ok=True)

        # For articles: use batch scoring to pre-select candidates
        if search_request.content_type != ContentType.IMAGES:
            # Batch score all candidates and select top ones (much more efficient!)
            desired_count = min(search_request.count * 3, len(candidates))  # Download 3x what we need
            candidates_to_download = self.search_engine.score_and_select_candidates(
                candidates, search_request, desired_count
            )
        else:
            # For images: process reasonable number (no description pre-filtering)
            candidates_to_download = candidates[:min(search_request.count * 2, len(candidates))]

        logger.info(f"📥 Downloading {len(candidates_to_download)} selected candidates")

        # Create temp directory for content scoring
        temp_dir = Path(tempfile.mkdtemp(prefix="scoring_"))
        scored_results = []

        try:
            for i, candidate in enumerate(candidates_to_download):
                logger.info(f"📥 Downloading {i+1}/{len(candidates_to_download)}: {candidate.title[:50]}...")

                # Download based on content type
                if search_request.content_type == ContentType.IMAGES:
                    result = self.downloader.download_image(candidate, directory)
                else:
                    result = self.downloader.download_webpage(candidate, directory)

                    # Extract text content for articles
                    if result.status == "success" and result.filepath.exists():
                        html_content = result.filepath.read_text(encoding='utf-8')
                        extracted_text = self.content_extractor.extract_article_text(html_content)
                        result.extracted_text = extracted_text

                        # Save extracted text as .txt file for easy validation
                        if extracted_text:
                            txt_filepath = result.filepath.with_suffix('.txt')
                            txt_filepath.write_text(extracted_text, encoding='utf-8')
                            logger.debug(f"💾 Saved extracted text: {txt_filepath.name}")

                # Set relevance score
                if result.status == "success":
                    if search_request.content_type == ContentType.IMAGES:
                        # Images need content-based scoring
                        score = self.search_engine.score_candidate(
                            candidate, search_request.content_type, search_request, temp_dir
                        )
                        result.relevance_score = score
                    else:
                        # Articles already have batch description score, use that
                        result.relevance_score = getattr(candidate, 'score', 0.0)
                else:
                    result.relevance_score = 0.0

                # Add search metadata
                result.search_description = getattr(candidate, 'description', None)
                candidate_title = getattr(candidate, 'title', None)
                if candidate_title and not result.title:
                    result.title = candidate_title

                scored_results.append(result)

        finally:
            # Clean up temp directory
            # try:
            #     shutil.rmtree(temp_dir)
            # except Exception as e:
            #     logger.warning(f"⚠️ Failed to clean up temp dir: {e}")
            pass

        # Sort by score and select best N
        successful_results = [r for r in scored_results if r.status == "success"]
        successful_results.sort(key=lambda x: x.relevance_score or 0.0, reverse=True)
        selected_results = successful_results[:search_request.count]

        # Clean up: move files that weren't selected to candidates subfolder
        selected_filepaths = {r.filepath for r in selected_results}
        unselected_results = [r for r in successful_results if r.filepath not in selected_filepaths]

        if unselected_results:
            candidates_dir = directory / "candidates"
            candidates_dir.mkdir(exist_ok=True)

            for result in unselected_results:
                try:
                    # Move HTML file
                    if result.filepath.exists():
                        new_html_path = candidates_dir / result.filepath.name
                        result.filepath.rename(new_html_path)
                        logger.debug(f"📁 Moved unselected file to: candidates/{result.filepath.name}")

                        # Also move corresponding .txt file if it exists
                        txt_filepath = result.filepath.with_suffix('.txt')
                        if txt_filepath.exists():
                            new_txt_path = candidates_dir / txt_filepath.name
                            txt_filepath.rename(new_txt_path)
                            logger.debug(f"📁 Moved extracted text to: candidates/{txt_filepath.name}")

                except Exception as e:
                    logger.warning(f"⚠️ Failed to move {result.filepath}: {e}")

        logger.info(f"✅ Download phase complete: {len(selected_results)} best results selected → {directory}")
        return selected_results

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
    parser.add_argument('--searxng-url', default='http://localhost:8080',
                        help='SearXNG instance URL')
    args = parser.parse_args()

    api_key = getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("❌ OPENROUTER_API_KEY not set. Get one from https://openrouter.ai/keys")
        return

    # Create agent and execute request
    try:
        agent = InternetSearchAgent(api_key, args.searxng_url)
        results = agent.execute_request(args.prompt)

        # Show summary
        successful = len([r for r in results if r.status == "success"])
        logger.info(f"🎉 Complete! {successful}/{len(results)} downloads successful")

    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
