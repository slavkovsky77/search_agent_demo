"""
Clean, modular AI agent for autonomous internet search and downloads using OpenAI function calling.
"""
import json
from pathlib import Path
from typing import List
from openai import OpenAI

from .models import (
    SearchRequest,
    DownloadResult,
    SearchCandidate,
    SearchAction,
    ContentType)
from .agent_tools import get_search_tools
from .search_engine import SearchEngine
from .file_downloader import FileDownloader
from .content_extractor import ContentExtractor
from .config import setup_logging, LLMModels, SystemConstants

logger = setup_logging(__name__)


class InternetSearchAgentV3:
    """
    Function calling version of the Internet Search Agent.
    Uses OpenAI's function calling instead of JSON parsing for better reliability.
    """

    def __init__(self, openrouter_api_key: str, searxng_url: str = SystemConstants.DEFAULT_SEARXNG_URL):
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/rapid-sos-search",
                "X-Title": "Internet Search Agent v3 (Function Calling)"
            }
        )

        # Initialize components
        self.search_engine = SearchEngine(searxng_url, self.client)
        self.download_dir = Path("downloads")
        self.downloader = FileDownloader(self.download_dir)
        self.content_extractor = ContentExtractor(self.client)

        # Get tool definitions from agent_tools module
        self.tools = get_search_tools()

        logger.info("🚀 InternetSearchAgent v3 (Function Calling) ready")

    def execute_request(self, user_request: str) -> List[DownloadResult]:
        """
        Execute user request using function calling.
        The AI decides which tools to use and how to use them.
        """
        logger.info(f"📝 Processing request with function calling: {user_request}")

        system_message = """You are an expert internet search and download assistant.

        Analyze the user's request and use the appropriate tools to fulfill it. You have access to:
        - search_images: Find and download images
        - download_articles: Find and download articles/webpages
        - download_webpage: Download a specific URL
        - analyze_content: Analyze downloaded content

        Guidelines:
        - For image requests like "find photos of X", use search_images
        - For article requests like "download articles about X", use download_articles
        - For specific URLs, use download_webpage
        - Extract numbers carefully (e.g., "3 photos" = count 3)
        - If website mentioned (e.g., "from NASA"), set source parameter
        - For "random" requests, set topic="random"

        Call the appropriate function(s) based on the user's request."""

        try:
            response = self.client.chat.completions.create(
                model=LLMModels.REQUEST_UNDERSTANDING,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_request}
                ],
                tools=self.tools,
                tool_choice="auto",
                **LLMModels.REQUEST_PARAMS
            )

            # Process tool calls
            results = []
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    logger.info(f"🔧 Calling tool: {function_name} with args: {function_args}")

                    # Execute the appropriate function
                    if function_name == "search_images":
                        tool_results = self._execute_search_images(**function_args)
                    elif function_name == "download_articles":
                        tool_results = self._execute_download_articles(**function_args)
                    elif function_name == "download_webpage":
                        tool_results = self._execute_download_webpage(**function_args)
                    elif function_name == "analyze_content":
                        # This would return analysis text, not DownloadResult
                        analysis = self._execute_analyze_content(**function_args)
                        logger.info(f"📊 Analysis result: {analysis}")
                        continue
                    else:
                        logger.warning(f"⚠️ Unknown function: {function_name}")
                        continue

                    results.extend(tool_results)
            else:
                logger.warning("⚠️ No tool calls were made by the AI")

            return results

        except Exception as e:
            logger.exception(f"❌ Failed to execute request: {e}")
            return []

    def _execute_search_images(self, subject: str, count: int, source: str = "any") -> List[DownloadResult]:
        """Execute image search function."""
        search_request = SearchRequest(
            action=SearchAction.SEARCH_IMAGES,
            subject=subject,
            count=count,
            content_type=ContentType.IMAGES,
            source=source,
            topic=subject
        )

        return self._original_execute_logic(search_request)

    def _execute_download_articles(self, topic: str, count: int, source: str = "any") -> List[DownloadResult]:
        """Execute article download function."""
        search_request = SearchRequest(
            action=SearchAction.DOWNLOAD_ARTICLES,
            subject="articles",
            count=count,
            content_type=ContentType.ARTICLES,
            source=source,
            topic=topic
        )

        return self._original_execute_logic(search_request)

    def _execute_download_webpage(self, url: str) -> List[DownloadResult]:
        """Execute webpage download function."""
        search_request = SearchRequest(
            action=SearchAction.DOWNLOAD_WEBPAGE,
            subject=url,
            count=1,
            content_type=ContentType.WEBPAGE,
            source=url.split("//")[1].split("/")[0] if "//" in url else "unknown",
            topic="webpage"
        )

        return self._original_execute_logic(search_request)

    def _execute_analyze_content(self, content: str, question: str) -> str:
        """Execute content analysis function."""
        try:
            from .prompts import get_content_analysis_prompt
            prompt = get_content_analysis_prompt(content, question)

            response = self.client.chat.completions.create(
                model=LLMModels.CONTENT_SCORING,
                messages=[{"role": "user", "content": prompt}],
                **LLMModels.SCORING_PARAMS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error analyzing content: {str(e)}"

    def _original_execute_logic(self, search_request: SearchRequest) -> List[DownloadResult]:
        """
        Execute the original search/download logic.
        This preserves all the existing functionality while using function calling for request understanding.
        """
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

    # Copy all the original methods from InternetSearchAgent
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

    def _select_candidates_for_download(self, candidates: list, search_request: SearchRequest) -> list:
        """Select which candidates to download using smart diminishing returns."""
        count = search_request.count

        # Smart multiplier calculation with conservative diminishing returns
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
            self, candidates: list, directory: Path, search_request: SearchRequest) -> list[DownloadResult]:
        """Download all candidates and extract content."""
        results = []
        candidates_dir = directory / "candidates"
        candidates_dir.mkdir(exist_ok=True)

        for i, candidate in enumerate(candidates):
            logger.info(f"📥 {i+1}/{len(candidates)}: {candidate.title[:50]}...")

            if search_request.content_type == ContentType.IMAGES:
                result = self.downloader.download_image(candidate, candidates_dir)
                if result:
                    result.relevance_score = self.search_engine._score_single_image(
                        result, search_request.subject)
            else:
                result = self.downloader.download_webpage(candidate, candidates_dir)
                if result:
                    result.relevance_score = self.search_engine._score_single_article(
                        result, search_request)
                    result.search_description = candidate.description
                    result.title = candidate.title

            if result:
                results.append(result)

        return results

    def _select_best_results(self, results: list[DownloadResult], count: int, directory: Path) -> list[DownloadResult]:
        """Select best results and move them to final directory."""
        # Sort by relevance score (highest first)
        results.sort(key=lambda x: x.relevance_score or 0.0, reverse=True)

        # Select top results
        selected_results = results[:count]

        # Move selected files to main directory
        for result in selected_results:
            # Move from candidates/ to main directory
            new_path = directory / result.filename
            try:
                result.filepath.rename(new_path)
                result.filepath = new_path
            except Exception as e:
                logger.warning(f"Failed to move {result.filepath}: {e}")

        return selected_results

    def _save_results_log(self, results: list[DownloadResult]) -> None:
        """Save download results to log file."""
        if not results:
            return

        log_file = self.download_dir / "download_log.json"

        # Load existing logs
        existing_logs = []
        if log_file.exists():
            try:
                existing_logs = json.loads(log_file.read_text())
            except Exception:
                pass

        # Add new results
        for result in results:
            log_entry = {
                "url": str(result.url),
                "filename": result.filename,
                "size_bytes": result.size_bytes,
                "download_time": result.download_time.isoformat(),
                "relevance_score": result.relevance_score,
                "title": result.title
            }
            existing_logs.append(log_entry)

        # Save logs
        log_file.write_text(json.dumps(existing_logs, indent=2))

    def _save_search_candidates(self, candidates: list, search_request: SearchRequest) -> None:
        """Save search candidates for debugging."""
        candidates_file = self.download_dir / "search_candidates.json"

        candidates_data = {
            "search_request": search_request.model_dump(),
            "candidates": [
                {
                    "url": str(candidate.url),
                    "title": candidate.title,
                    "description": getattr(candidate, 'description', None),
                    "score": getattr(candidate, 'score', 0.0)
                }
                for candidate in candidates
            ]
        }

        candidates_file.write_text(json.dumps(candidates_data, indent=2))
