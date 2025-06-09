#!/usr/bin/env python3
"""
Integration tests for the Internet Search Agent
Tests actually run the agent and validate results with LLM
"""

import pytest
import json
import os
from pathlib import Path
from agent_search import InternetSearchAgent
from config import setup_logging

logger = setup_logging(__name__)


@pytest.fixture
def agent() -> InternetSearchAgent:
    """Create real agent instance for testing."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    searxng_url = os.getenv('SEARXNG_URL', 'http://localhost:8080')
    return InternetSearchAgent(api_key, searxng_url)


@pytest.fixture
def test_downloads_dir() -> Path:
    """Create clean test download directory in workspace."""
    import shutil

    test_dir = Path("test_downloads")

    # Clean up any existing test results at start
    if test_dir.exists():
        shutil.rmtree(test_dir)

    test_dir.mkdir(exist_ok=True)
    return test_dir


class TestImageDownloads:
    """Test actual image download functionality."""

    def test_download_elephant_photos(self, agent: InternetSearchAgent, test_downloads_dir: Path) -> None:
        """Test downloading elephant photos and validate they are actually elephants."""
        # Override download directory
        agent.download_dir = test_downloads_dir

        # Execute the request
        results = agent.execute_request("Find 2 photos of elephants")

        # Check we got exactly what we asked for
        successful_downloads = [r for r in results if r.get('status') == 'success']
        assert len(successful_downloads) >= 2, f"Should download 2 images, got {len(successful_downloads)}"

        # Validate ALL downloaded images
        validation_results = []
        for i, result in enumerate(successful_downloads[:2]):
            if 'filepath' in result:
                filepath = Path(result['filepath'])
                if filepath.exists():
                    logger.info(f"🔍 Validating image {i+1}/2: {filepath.name}")
                    is_elephant = self._validate_image_with_vision(agent, filepath, "elephants")
                    validation_results.append(is_elephant)

                    # Each image should be an elephant
                    assert is_elephant.get('relevant', False), f"Image {i+1} should be an elephant. Got: {is_elephant}"

        # Should have validated exactly 2 images
        assert len(validation_results) == 2, f"Should validate 2 images, validated {len(validation_results)}"
        logger.info("✅ All 2 elephant images validated successfully")

    def test_download_mountain_photos(self, agent: InternetSearchAgent, test_downloads_dir: Path) -> None:
        """Test downloading mountain photos and validate ALL are actually mountains."""
        agent.download_dir = test_downloads_dir

        # Execute the request
        results = agent.execute_request("Find 2 photos of mountains")

        # Check we got exactly what we asked for
        successful_downloads = [r for r in results if r.get('status') == 'success']
        assert len(successful_downloads) >= 2, f"Should download 2 images, got {len(successful_downloads)}"

        # Validate ALL downloaded images
        validation_results = []
        for i, result in enumerate(successful_downloads[:2]):
            if 'filepath' in result:
                filepath = Path(result['filepath'])
                if filepath.exists():
                    logger.info(f"🔍 Validating image {i+1}/2: {filepath.name}")
                    is_mountain = self._validate_image_with_vision(agent, filepath, "mountains")
                    validation_results.append(is_mountain)

                    # Each image should be a mountain
                    assert is_mountain.get('relevant', False), f"Image {i+1} should be a mountain. Got: {is_mountain}"

        # Should have validated exactly 2 images
        assert len(validation_results) == 2, f"Should validate 2 images, validated {len(validation_results)}"
        logger.info("✅ All 2 mountain images validated successfully")

    def _validate_image_with_vision(self, agent: InternetSearchAgent, filepath: Path, expected_subject: str) -> dict:
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
                "confidence": <0.0-1.0,
                "reason": "detailed description of what you see and why it matches/doesn't match {expected_subject}"
            }}
            """

            response = agent.client.chat.completions.create(
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
            logger.info(f"🔍 Vision validation for {filepath.name}: {result}")
            return result

        except Exception as e:
            logger.warning(f"❌ Vision validation failed for {filepath.name}: {e}")
            return {"relevant": False, "confidence": 0.0, "reason": f"Vision validation failed: {e}"}


class TestArticleDownloads:
    """Test actual article download functionality."""

    def test_download_wikipedia_articles(self, agent: InternetSearchAgent, test_downloads_dir: Path) -> None:
        """Test downloading Wikipedia articles about a specific topic and validate both source AND topic."""
        agent.download_dir = test_downloads_dir

        results = agent.execute_request("Download 2 wikipedia articles about ancient rome")

        # Check we got exactly what we asked for
        successful_downloads = [r for r in results if r.get('status') == 'success']
        logger.info(f"📊 Downloaded {len(successful_downloads)} Rome articles successfully")
        assert len(successful_downloads) >= 2, f"Should download 2 articles, got {len(successful_downloads)}"

        # Validate ALL downloaded articles for BOTH source and topic
        validation_results = []
        for i, result in enumerate(successful_downloads[:2]):
            if 'filepath' in result:
                filepath = Path(result['filepath'])
                if filepath.exists():
                    logger.info(f"🔍 Validating Rome article {i+1}/2: {filepath.name}")
                    content = filepath.read_text(encoding='utf-8')

                    # Check BOTH: Wikipedia source AND Rome topic
                    is_wikipedia_and_rome = self._validate_wikipedia_topic_content(
                        agent, content, "ancient rome"
                    )
                    logger.info(f"📝 Wikipedia+Rome validation for {filepath.name}: {is_wikipedia_and_rome}")
                    validation_results.append(is_wikipedia_and_rome)

                    # Each article should be Wikipedia AND about Rome
                    assert is_wikipedia_and_rome.get('relevant', False), (
                        f"Article {i+1} should be Wikipedia about Rome. "
                        f"Got: {is_wikipedia_and_rome}"
                    )

        # Should have validated exactly 2 articles
        assert len(validation_results) == 2, f"Should validate 2 articles, validated {len(validation_results)}"
        logger.info("✅ All 2 Wikipedia Rome articles validated successfully")

    def test_download_ai_articles(self, agent: InternetSearchAgent, test_downloads_dir: Path) -> None:
        """Test downloading articles about AI and validate ALL are about AI."""
        agent.download_dir = test_downloads_dir

        results = agent.execute_request("Download 2 articles about artificial intelligence")

        # Check we got exactly what we asked for
        successful_downloads = [r for r in results if r.get('status') == 'success']
        logger.info(f"📊 Downloaded {len(successful_downloads)} AI articles successfully")
        assert len(successful_downloads) >= 2, f"Should download 2 AI articles, got {len(successful_downloads)}"

        # Validate ALL downloaded articles
        validation_results = []
        for i, result in enumerate(successful_downloads[:2]):
            if 'filepath' in result:
                filepath = Path(result['filepath'])
                if filepath.exists():
                    logger.info(f"🔍 Validating AI article {i+1}/2: {filepath.name}")
                    content = filepath.read_text(encoding='utf-8')

                    # Check if it's about AI (topic validation)
                    is_ai_related = self._validate_article_content(
                        agent, content, "artificial intelligence"
                    )
                    logger.info(f"📝 AI topic validation for {filepath.name}: {is_ai_related}")
                    validation_results.append(is_ai_related)

                    # Each article should be about AI
                    assert is_ai_related.get('relevant', False), (
                        f"Article {i+1} should be about AI. Got: {is_ai_related}"
                    )

        # Should have validated exactly 2 articles
        assert len(validation_results) == 2, f"Should validate 2 articles, validated {len(validation_results)}"
        logger.info("✅ All 2 AI articles validated successfully")

    def test_download_specific_site_articles(self, agent: InternetSearchAgent, test_downloads_dir: Path) -> None:
        """Test downloading articles from specific sites and validate URLs."""
        agent.download_dir = test_downloads_dir

        results = agent.execute_request("Download 1 article about space from nasa.gov")

        successful_downloads = [r for r in results if r.get('status') == 'success']
        logger.info(f"📊 Downloaded {len(successful_downloads)} NASA articles successfully")
        assert len(successful_downloads) >= 1, "Should download at least 1 NASA article"

        # Check URL contains nasa.gov
        for i, result in enumerate(successful_downloads[:1]):
            url = result.get('url', '')
            logger.info(f"🔍 Checking NASA URL {i+1}: {url}")
            assert 'nasa.gov' in url.lower(), f"Should be from NASA.gov, got: {url}"

        logger.info("✅ NASA URL validation successful")

    def _validate_wikipedia_topic_content(self, agent: InternetSearchAgent, content: str, topic: str) -> dict:
        """Validate that content is from Wikipedia AND about the specified topic."""
        # Use more content for better validation (first 5000 chars)
        text_snippet = self._extract_text_snippet(content, max_chars=5000)

        validation_prompt = f"""
        Content snippet: {text_snippet}
        Expected topic: {topic}

        Is this content from Wikipedia AND about {topic}? Check BOTH:
        1. Wikipedia source (look for Wikipedia formatting, references, categories)
        2. Topic relevance (does the content discuss {topic}?)

        Return ONLY valid JSON in this exact format:
        {{
            "relevant": true/false,
            "confidence": <0.0-1.0>,
            "reason": "explanation covering both Wikipedia source and topic relevance"
        }}

        Do not include any other text, only the JSON."""

        return self._get_llm_validation(agent, validation_prompt)

    def _validate_article_content(self, agent: InternetSearchAgent, content: str, topic: str) -> dict:
        """Validate that article content matches the expected topic."""
        text_snippet = self._extract_text_snippet(content, max_chars=5000)

        validation_prompt = f"""
        Expected topic: {topic}
        Article content: {text_snippet}

        Is this article about {topic}? Analyze the content for relevance.

        Return ONLY valid JSON in this exact format:
        {{
            "relevant": true/false,
            "confidence": <0.0-1.0>,
            "reason": "explanation"
        }}

        Do not include any other text, only the JSON."""

        return self._get_llm_validation(agent, validation_prompt)

    def _extract_text_snippet(self, html_content: str, max_chars: int = 800) -> str:
        """Extract readable text from HTML content."""
        import re
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        # Clean up whitespace
        text = ' '.join(text.split())
        # Return specified number of characters
        return text[:max_chars]

    def _get_llm_validation(self, agent: InternetSearchAgent, prompt: str) -> dict:
        """Get validation response from LLM."""
        try:
            # logger.debug(f"🤖 Validating with prompt: {prompt}")
            response = agent.client.chat.completions.create(
                model="anthropic/claude-3-sonnet",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )

            content = response.choices[0].message.content.strip()
            logger.debug(f"🤖 Raw LLM validation response: {content}")

            result = json.loads(content)
            logger.debug(f"🔍 Parsed validation result: {result}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}, raw content: {content}")
            return {"relevant": False, "confidence": 0.0, "reason": f"JSON parsing failed: {content[:100]}"}
        except Exception as e:
            logger.error(f"❌ LLM validation failed: {e}")
            return {"relevant": False, "confidence": 0.0, "reason": f"Validation failed: {e}"}
