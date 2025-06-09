#!/usr/bin/env python3
"""
Integration tests for the Internet Search Agent
Tests actually run the agent and validate results with LLM
"""

import pytest
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from agent_search import InternetSearchAgent
from config import setup_logging
import trafilatura

logger = setup_logging(__name__)


@dataclass
class TestScenario:
    """Test scenario configuration."""
    name: str
    request: str
    expected_count: int
    content_type: str  # "images", "articles", "webpage"
    expected_topic: Optional[str] = None
    expected_source: Optional[str] = None
    url_validation: Optional[str] = None


@pytest.fixture
def agent() -> InternetSearchAgent:
    """Create real agent instance for testing."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    searxng_url = os.getenv('SEARXNG_URL', 'http://localhost:8080')
    return InternetSearchAgent(api_key, searxng_url)


@pytest.fixture
def test_downloads_dir(scope="session") -> Path:
    """Create clean test download directory in workspace."""
    import shutil

    test_dir = Path("test_downloads")

    # Clean up any existing test results at start
    if test_dir.exists():
        shutil.rmtree(test_dir)

    test_dir.mkdir(exist_ok=True)
    return test_dir


class ContentValidator:
    """General content validator for images, articles, and webpages."""

    def __init__(self, agent: InternetSearchAgent):
        self.agent = agent

    def validate_downloads(self, results: List[dict], scenario: TestScenario) -> None:
        """Validate all downloads match the scenario expectations."""
        successful_downloads = [r for r in results if r.get('status') == 'success']
        logger.info(f"📊 Downloaded {len(successful_downloads)} files successfully")

        # Check count
        assert len(successful_downloads) >= scenario.expected_count, (
            f"Should download {scenario.expected_count} items, got {len(successful_downloads)}"
        )

        # Validate each download
        validation_results = []
        for i, result in enumerate(successful_downloads[:scenario.expected_count]):
            validation_result = self._validate_single_download(result, scenario, i + 1)
            validation_results.append(validation_result)

        # All should pass validation
        passed_count = sum(1 for v in validation_results if v)
        assert passed_count == scenario.expected_count, (
            f"Should validate {scenario.expected_count} items, {passed_count} passed"
        )

        logger.info(f"✅ All {scenario.expected_count} {scenario.content_type} validated successfully")

    def _validate_single_download(self, result: dict, scenario: TestScenario, item_num: int) -> bool:
        """Validate a single download result."""
        if scenario.content_type == "images":
            return self._validate_image(result, scenario, item_num)
        elif scenario.content_type in ["articles", "webpage"]:
            return self._validate_article(result, scenario, item_num)
        else:
            logger.warning(f"Unknown content type: {scenario.content_type}")
            return False

    def _validate_image(self, result: dict, scenario: TestScenario, item_num: int) -> bool:
        """Validate image download using vision AI."""
        if 'filepath' not in result:
            return False

        filepath = Path(result['filepath'])
        if not filepath.exists():
            return False

        logger.info(f"🔍 Validating image {item_num}/{scenario.expected_count}: {filepath.name}")

        # Use vision validation if topic specified
        if scenario.expected_topic:
            validation = self._validate_image_with_vision(filepath, scenario.expected_topic)
            logger.info(f"📝 Vision validation: {validation}")

            is_valid = validation.get('relevant', False)
            assert is_valid, f"Image {item_num} should show {scenario.expected_topic}. Got: {validation}"
            return is_valid

        return True  # No topic validation needed

    def _validate_article(self, result: dict, scenario: TestScenario, item_num: int) -> bool:
        """Validate article/webpage download."""
        # URL validation first
        if scenario.url_validation:
            url = result.get('url', '')
            logger.info(f"🔍 Checking URL {item_num}: {url}")
            is_url_valid = scenario.url_validation.lower() in url.lower()
            assert is_url_valid, f"URL should contain {scenario.url_validation}, got: {url}"
            if not scenario.expected_topic:  # Only URL validation needed
                return True

        # Content validation
        if 'filepath' not in result:
            return False

        filepath = Path(result['filepath'])
        if not filepath.exists():
            return False

        logger.info(f"🔍 Validating article {item_num}/{scenario.expected_count}: {filepath.name}")

        content = filepath.read_text(encoding='utf-8')

        # Choose validation type
        if scenario.expected_source and scenario.expected_topic:
            validation = self._validate_source_and_topic(content, scenario.expected_source, scenario.expected_topic)
        elif scenario.expected_topic:
            validation = self._validate_topic_only(content, scenario.expected_topic)
        elif scenario.expected_source:
            validation = self._validate_source_only(content, scenario.expected_source)
        else:
            # Basic validation for random articles - ensure we got real content
            validation = self._validate_basic_article_content(content)

        logger.info(f"📝 Content validation: {validation}")
        is_valid = validation.get('relevant', False)
        assert is_valid, f"Article {item_num} validation failed. Got: {validation}"
        return is_valid

    def _validate_image_with_vision(self, filepath: Path, expected_subject: str) -> dict:
        """Use vision LLM to validate actual image content."""
        import base64

        try:
            # Read and encode image
            image_data = filepath.read_bytes()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Get file extension for mime type
            ext = filepath.suffix.lower()
            mime_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }.get(ext, 'image/jpeg')

            validation_prompt = f"""
            Look at this image and determine if it shows {expected_subject}.

            Analyze the visual content carefully.

            Return JSON:
            {{
                "relevant": true/false,
                "confidence": <0.0-1.0>,
                "reason": "detailed description of what you see and why it matches/doesn't match {expected_subject}"
            }}
            """

            response = self.agent.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": validation_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )

            result = json.loads(response.choices[0].message.content.strip())
            return result

        except Exception as e:
            logger.warning(f"❌ Vision validation failed for {filepath.name}: {e}")
            return {"relevant": False, "confidence": 0.0, "reason": f"Vision validation failed: {e}"}

    def _validate_source_and_topic(self, content: str, source: str, topic: str) -> dict:
        """Validate both source and topic."""
        text_snippet = self._extract_text_snippet(content, max_chars=5000)

        validation_prompt = f"""
        Content snippet: {text_snippet}
        Expected source: {source}
        Expected topic: {topic}

        Is this content from {source} AND about {topic}? Check BOTH:
        1. Source verification (look for {source} indicators)
        2. Topic relevance (does the content discuss {topic}?)

        Return ONLY valid JSON:
        {{
            "relevant": true/false,
            "confidence": <0.0-1.0>,
            "reason": "explanation covering both source and topic"
        }}"""

        return self._get_llm_validation(validation_prompt)

    def _validate_topic_only(self, content: str, topic: str) -> dict:
        """Validate topic relevance only."""
        text_snippet = self._extract_text_snippet(content, max_chars=5000)

        validation_prompt = f"""
        Content snippet: {text_snippet}
        Expected topic: {topic}

        Is this content about {topic}? Analyze for topic relevance.

        Return ONLY valid JSON:
        {{
            "relevant": true/false,
            "confidence": <0.0-1.0>,
            "reason": "explanation of topic relevance"
        }}"""

        return self._get_llm_validation(validation_prompt)

    def _validate_source_only(self, content: str, source: str) -> dict:
        """Validate source only."""
        text_snippet = self._extract_text_snippet(content, max_chars=5000)

        validation_prompt = f"""
        Content snippet: {text_snippet}
        Expected source: {source}

        Is this content from {source}? Look for {source} indicators.

        Return ONLY valid JSON:
        {{
            "relevant": true/false,
            "confidence": <0.0-1.0>,
            "reason": "explanation of source validation"
        }}"""

        return self._get_llm_validation(validation_prompt)

    def _validate_basic_article_content(self, content: str) -> dict:
        """Basic validation for random articles - ensure we got real content."""
        text_snippet = self._extract_text_snippet(content, max_chars=3000)

        validation_prompt = f"""
        Content snippet: {text_snippet}

        Is this real article content (not an error page, empty page, or placeholder)?

        Check for:
        - Substantial text content (not just navigation/headers)
        - Article-like structure with paragraphs
        - Not error messages like "404", "Page not found", "Access denied"
        - Not just menus, ads, or boilerplate text

        Return ONLY valid JSON:
        {{
            "relevant": true/false,
            "confidence": <0.0-1.0>,
            "reason": "explanation of whether this is real article content"
        }}"""

        return self._get_llm_validation(validation_prompt)

    def _extract_text_snippet(self, html_content: str, max_chars: int = 5000) -> str:
        """Extract readable article content using smart extraction libraries."""
        try:
            # Try trafilatura first - purpose-built for article extraction

            extracted = trafilatura.extract(html_content, include_comments=False, include_tables=False)
            if extracted and len(extracted.strip()) > 100:
                logger.debug(f"🔍 Trafilatura extracted {len(extracted)} chars of clean content")
                return extracted[:max_chars]

        except ImportError:
            logger.debug("📦 Trafilatura not available, trying LLM extraction")
        except Exception as e:
            logger.debug(f"⚠️ Trafilatura failed: {e}, trying LLM extraction")

        # Fallback: Use LLM to extract main content
        try:
            return self._extract_with_llm(html_content, max_chars)
        except Exception as e:
            logger.exception(f"❌ LLM extraction failed: {e}, using regex fallback")
            raise e

    def _extract_with_llm(self, html_content: str, max_chars: int) -> str:
        """Use LLM to extract main article content from HTML."""
        # Send a chunk of HTML to LLM for smart extraction
        html_chunk = html_content[:15000]  # Send first 15k chars to avoid token limits

        extraction_prompt = f"""
        Extract the main article content from this HTML, removing navigation, ads, scripts, and boilerplate.
        Return ONLY the readable article text, not JSON or explanations.

        HTML:
        {html_chunk}

        Return only the clean article text:"""

        response = self.agent.client.chat.completions.create(
            model="anthropic/claude-3-haiku",  # Faster/cheaper model for extraction
            messages=[{"role": "user", "content": extraction_prompt}],
            max_tokens=2000,
            temperature=0.1
        )

        extracted_text = response.choices[0].message.content.strip()
        logger.debug(f"🤖 LLM extracted {len(extracted_text)} chars of content")

        if len(extracted_text) > 100:
            return extracted_text[:max_chars]
        else:
            raise Exception("LLM extraction returned insufficient content")

    def _get_llm_validation(self, prompt: str) -> dict:
        """Get validation response from LLM."""
        try:
            logger.debug(f"🤖 Validating with prompt: {prompt}")
            response = self.agent.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()
            logger.debug(f"🤖 Raw LLM validation response: {content}")

            result = json.loads(content)
            logger.info(f"🔍 Parsed validation result: {result}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}, raw content: {content}")
            return {"relevant": False, "confidence": 0.0, "reason": f"JSON parsing failed: {content[:100]}"}
        except Exception as e:
            logger.error(f"❌ LLM validation failed: {e}")
            return {"relevant": False, "confidence": 0.0, "reason": f"Validation failed: {e}"}


class TestScenarios:
    """Test all scenarios from launch.json configurations."""

    # Define all test scenarios based on launch.json
    SCENARIOS = [
        TestScenario(
            name="elephant_photos",
            request="Find 2 photos of elephants",
            expected_count=2,
            content_type="images",
            expected_topic="elephants"
        ),
        TestScenario(
            name="mountain_photos",
            request="Find 3 photos of mountains",
            expected_count=3,
            content_type="images",
            expected_topic="mountains"
        ),
        TestScenario(
            name="random_articles",
            request="Download 2 random articles",
            expected_count=2,
            content_type="articles"
        ),
        TestScenario(
            name="ai_articles",
            request="Download 2 articles about artificial intelligence",
            expected_count=2,
            content_type="articles",
            expected_topic="artificial intelligence"
        ),
        TestScenario(
            name="tech_news",
            request="Download 2 news articles about technology",
            expected_count=2,
            content_type="articles",
            expected_topic="technology"
        ),
        TestScenario(
            name="nasa_articles",
            request="Download 2 articles about space from nasa.gov",
            expected_count=2,
            content_type="articles",
            expected_topic="space",
            url_validation="nasa.gov"
        ),
        TestScenario(
            name="wikipedia_random",
            request="download 2 random wikipedia articles",
            expected_count=2,
            content_type="articles",
            expected_source="wikipedia.org",
        ),
        TestScenario(
            name="wikipedia_punic_wars",
            request="download 2 wikipedia articles about punic wars",
            expected_count=2,
            content_type="articles",
            expected_source="wikipedia.org",
            expected_topic="punic wars"
        ),
        TestScenario(
            name="cnn_ukraine",
            request="download 2 cnn articles about ukraine war",
            expected_count=2,
            content_type="articles",
            expected_source="cnn.com",
            expected_topic="ukraine war"
        ),
        TestScenario(
            name="business_news",
            request="Download latest business news",
            expected_count=1,
            content_type="articles",
            expected_topic="business"
        )
    ]

    @pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
    def test_scenario(self, scenario: TestScenario, agent: InternetSearchAgent, test_downloads_dir: Path) -> None:
        """Test a specific scenario from launch.json."""
        agent.download_dir = test_downloads_dir
        validator = ContentValidator(agent)

        logger.info(f"🎯 Testing scenario: {scenario.name}")
        logger.info(f"📝 Request: {scenario.request}")

        # Execute the request
        results = agent.execute_request(scenario.request)

        # Validate results
        validator.validate_downloads(results, scenario)

        logger.info(f"🎉 Scenario {scenario.name} completed successfully!")
