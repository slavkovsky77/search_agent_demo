"""
Content extraction from HTML articles using trafilatura and LLM fallback.
"""
from openai import OpenAI
import trafilatura

from . import prompts
from .config import setup_logging

logger = setup_logging(__name__)


class ContentExtractor:
    """Extracts readable text content from HTML articles."""

    def __init__(self, openai_client: OpenAI | None = None):
        self.client = openai_client

    def extract_article_text(self, html_content: str, max_chars: int = 5000) -> str:
        """Extract main article text from HTML using smart extraction."""
        # Try trafilatura first (purpose-built for article extraction)
        try:
            extracted = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=False
            )

            if extracted and len(extracted.strip()) > 100:
                logger.debug(f"🔍 Trafilatura extracted {len(extracted)} chars of clean content")
                return extracted[:max_chars]

        except Exception as e:
            logger.debug(f"⚠️ Trafilatura failed: {e}")

        # Fallback to LLM extraction if trafilatura fails
        if self.client:
            try:
                return self._extract_with_llm(html_content, max_chars)
            except Exception as e:
                logger.exception(f"❌ LLM extraction failed: {e}")

        # Last resort: basic regex cleanup
        return self._extract_basic_text(html_content, max_chars)

    def _extract_with_llm(self, html_content: str, max_chars: int) -> str:
        """Use LLM to extract main article content from HTML."""
        prompt = prompts.get_text_extraction_prompt(html_content)

        response = self.client.chat.completions.create(
            model="anthropic/claude-3-haiku",  # Faster/cheaper model for extraction
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1
        )

        extracted_text = response.choices[0].message.content.strip()
        logger.debug(f"🤖 LLM extracted {len(extracted_text)} chars of content")

        if len(extracted_text) > 100:
            return extracted_text[:max_chars]
        else:
            raise Exception("LLM extraction returned insufficient content")

    def _extract_basic_text(self, html_content: str, max_chars: int) -> str:
        """Basic text extraction as last resort."""
        import re

        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)

        # Clean up whitespace
        text = ' '.join(text.split())

        logger.debug(f"📝 Basic extraction got {len(text)} chars")
        return text[:max_chars]
