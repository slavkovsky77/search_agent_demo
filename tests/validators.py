"""
Content validation for downloaded files using LLM validation.
"""
import json
import base64
from pathlib import Path

from src.content_extractor import ContentExtractor
from src.config import setup_logging
from src.models import DownloadResult
from .scenarios import TestScenario

logger = setup_logging(__name__)


class ContentValidator:
    """General content validator for images, articles, and webpages."""

    def __init__(self, agent):
        self.agent = agent
        self.content_extractor = ContentExtractor(agent.client)

    def validate_downloads(self, results: list[DownloadResult], scenario: TestScenario) -> None:
        """Validate all downloads match the scenario expectations."""
        successful_downloads = [r for r in results if r.status == 'success']
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

    def _validate_single_download(self, result: DownloadResult, scenario: TestScenario, item_num: int) -> bool:
        """Validate a single download result."""
        if scenario.content_type == "images":
            return self._validate_image(result, scenario, item_num)
        elif scenario.content_type in ["articles", "webpage"]:
            return self._validate_article(result, scenario, item_num)
        else:
            logger.warning(f"Unknown content type: {scenario.content_type}")
            return False

    def _validate_image(self, result: DownloadResult, scenario: TestScenario, item_num: int) -> bool:
        """Validate image download using vision AI."""
        if not result.filepath.exists():
            return False

        logger.info(f"🔍 Validating image {item_num}/{scenario.expected_count}: {result.filepath.name}")

        # Use vision validation if topic specified
        if scenario.expected_topic:
            validation = self._validate_image_with_vision(result.filepath, scenario.expected_topic, result)
            logger.info(f"📝 Vision validation: {validation}")

            is_valid = validation.get('relevant', False)
            assert is_valid, f"Image {item_num} should show {scenario.expected_topic}. Got: {validation}"
            return is_valid

        return True  # No topic validation needed

    def _validate_article(self, result: DownloadResult, scenario: TestScenario, item_num: int) -> bool:
        """Validate article/webpage download."""
        # URL validation first
        if scenario.url_validation:
            url = str(result.url)
            logger.info(f"🔍 Checking URL {item_num}: {url}")
            is_url_valid = scenario.url_validation.lower() in url.lower()
            assert is_url_valid, f"URL should contain {scenario.url_validation}, got: {url}"
            if not scenario.expected_topic:  # Only URL validation needed
                return True

        # Content validation
        if not result.filepath.exists():
            return False

        logger.info(f"🔍 Validating article {item_num}/{scenario.expected_count}: {result.filepath.name}")

        content = result.filepath.read_text(encoding='utf-8')

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

    def _validate_image_with_vision(self, filepath: Path, expected_subject: str, result: DownloadResult) -> dict:
        """Use vision LLM to validate actual image content."""
        try:
            # Read and encode image
            image_data = filepath.read_bytes()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Get actual image format from file content or metadata
            # Use the image_format from download result if available
            if hasattr(result, 'image_format') and result.image_format:
                format_lower = result.image_format.lower()
                mime_type = {
                    'jpeg': 'image/jpeg',
                    'jpg': 'image/jpeg',
                    'png': 'image/png',
                    'gif': 'image/gif',
                    'webp': 'image/webp'
                }.get(format_lower, 'image/jpeg')
            else:
                # Fallback to extension-based detection
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
        text_snippet = self.content_extractor.extract_article_text(content, max_chars=5000)

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
        text_snippet = self.content_extractor.extract_article_text(content, max_chars=5000)

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
        text_snippet = self.content_extractor.extract_article_text(content, max_chars=5000)

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
        text_snippet = self.content_extractor.extract_article_text(content, max_chars=3000)

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
