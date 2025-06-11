"""
Content extraction from HTML articles using trafilatura and LLM fallback.
"""
from openai import OpenAI
from pathlib import Path
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
        return None

    def save_extracted_text(self, extracted_text: str, html_filepath: Path) -> bool:
        """Save extracted text as .txt file alongside the HTML file."""
        try:
            txt_filepath = html_filepath.with_suffix('.txt')
            txt_filepath.write_text(extracted_text, encoding='utf-8')
            logger.debug(f"✅ Saved {len(extracted_text)} chars to {txt_filepath.name}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to save text for {html_filepath.name}: {e}")
            return False

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
