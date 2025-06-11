"""
Centralized prompt functions for the Internet Search Agent.
All AI prompts are defined here for better maintainability and reusability.
"""
from .models import SearchRequest


def get_request_understanding_prompt() -> str:
    """Prompt for understanding user requests."""
    return """You are an expert at understanding user requests for downloading content from the internet.

    Analyze the user's request and return EXACTLY this JSON format:
    {
        "action": "search_images" | "download_articles" | "download_webpage",
        "subject": "what to search for or download",
        "count": number_of_items,
        "content_type": "images" | "articles" | "webpage",
        "source": "specific website or 'any'",
        "topic": "specific topic or 'random'"
    }

    Examples:
    - "Find photos" → {"action": "search_images", "subject": "Y", "count": X,
      "content_type": "images", "source": "any", "topic": "Y"}
    - "Download random articles" → {"action": "download_articles", "subject": "articles",
      "count": N, "content_type": "articles", "source": "any", "topic": "random"}
    - "Download wikipedia articles" → {"action": "download_articles", "subject": "articles",
      "count": N, "content_type": "articles", "source": "wikipedia.org", "topic": "random"}
    - "Download https://example.com" → {"action": "download_webpage", "subject": "https://example.com",
      "count": 1, "content_type": "webpage", "source": "example.com", "topic": "webpage"}

    IMPORTANT: Extract the source website correctly!
    - "wikipedia articles" → source: "wikipedia.org"
    - "articles from BBC" → source: "bbc.com"
    - If no specific source mentioned → source: "any"

    Return ONLY the JSON, no other text."""


def get_search_queries_prompt(subject: str, count: int) -> str:
    """Prompt for generating search queries."""
    return f"""Generate {min(count, 5)} different search queries to find images of: {subject}

    Return EXACTLY a JSON array like: ["query1", "query2", "query{count}"]
    Each query should be 1-4 words, designed to find {subject}.
    Return ONLY the JSON array."""


def get_content_queries_prompt(topic: str, source: str, content_type: str, count: int) -> str:
    """Prompt for generating content search queries."""
    return f"""Generate {min(count, 3)} search queries to find {content_type}.

    Topic: {topic}
    Source: {source}

    Instructions:
    - If topic is "random": generate diverse interesting topics (history, science, culture, etc.)
    - If topic is specific: use that topic for all queries
    - If source is "any": DO NOT add any site: constraints
    - If source is specific website: add "site:{source}" to EVERY query
    - Never use the word "random" in queries
    - Make queries varied and effective for finding good {content_type}

    Examples:
    - Topic "random", Source "any" → ["history", "science discoveries", "cultural traditions"]
    - Topic "random", Source "wikipedia.org" → ["history site:wikipedia.org", "science site:wikipedia.org"]
    - Topic "AI", Source "any" → ["artificial intelligence", "AI technology", "machine learning"]
    - Topic "AI", Source "cnn.com" → ["AI site:cnn.com", "artificial intelligence site:cnn.com"]

    Return EXACTLY a JSON array of {min(count, 3)} search queries:
    ["query1", "query2", "query3"]

    Return ONLY the JSON array."""


def get_article_scoring_prompt(candidates: list[str], search_request: SearchRequest) -> str:
    """Prompt for scoring search candidates."""
    candidates_text = "\n".join([
        f"{i+1}. {candidate}" for i, candidate in enumerate(candidates)
    ])

    return f"""Score these search results for relevance to: "{search_request.subject}"
    Target: {search_request.count} {search_request.content_type.value} about "{search_request.topic}"
    Source: {search_request.source}

    Candidates:
    {candidates_text}

    Rate each candidate from 0.0 to 1.0 based on:
    - Topic relevance to "{search_request.topic}"
    - Source match with "{search_request.source}"
    - Content quality indicators
    - Title/description quality

    Return EXACTLY a JSON array of scores: [0.8, 0.6, 0.9, ...]
    Return ONLY the JSON array of scores."""


def get_text_extraction_prompt(html_content: str) -> str:
    """Prompt for extracting article text from HTML."""
    return f"""Extract the main article content from this HTML, removing navigation, ads, scripts, and boilerplate.
    Return ONLY the readable article text, not JSON or explanations.

    HTML:
    {html_content[:15000]}

    Return only the clean article text:"""


def get_image_scoring_prompt(subject: str) -> str:
    """Prompt for scoring images with vision LLM."""
    return f"""Look at this image and rate how well it shows {subject}.

    Rate the image based on:
    - Clarity and quality
    - Relevance to {subject}
    - Visual appeal
    - Whether it actually shows {subject}

    Return ONLY a JSON with a score 0.0-1.0:
    {{"score": 0.85, "reason": "clear photo of {subject}"}}"""
