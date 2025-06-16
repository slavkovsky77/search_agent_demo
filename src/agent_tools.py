"""
Function calling tool definitions for the Internet Search Agent.
Defines the structured schemas for OpenAI function calling.
"""
from typing import List, Dict, Any
from .config import SystemConstants, QueryLimits


def get_search_tools() -> List[Dict[str, Any]]:
    """Define available tools for OpenAI function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_images",
                "description": "Search for and download images based on a subject/topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "What images to search for (e.g., 'zebras', 'mountains', 'cats')"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of images to download",
                            "minimum": 1,
                            "maximum": SystemConstants.MAX_IMAGE_CANDIDATES
                        },
                        "source": {
                            "type": "string",
                            "description": "Specific website to search (optional)",
                            "default": "any"
                        }
                    },
                    "required": ["subject", "count"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "download_articles",
                "description": "Search for and download articles/webpages about a topic",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Topic to search for articles (use 'random' for diverse topics)"
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of articles to download",
                            "minimum": 1,
                            "maximum": QueryLimits.MAX_CONTENT_QUERIES * 5  # Allow more articles than queries
                        },
                        "source": {
                            "type": "string",
                            "description": "Specific website to search or 'any'",
                            "default": "any"
                        }
                    },
                    "required": ["topic", "count"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "download_webpage",
                "description": "Download a specific webpage by URL",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to download"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_content",
                "description": "Analyze downloaded content to answer questions about it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to analyze",
                            "maxLength": SystemConstants.DEFAULT_MAX_TEXT_CHARS
                        },
                        "question": {
                            "type": "string",
                            "description": "Question to answer about the content",
                            "maxLength": 500  # Reasonable limit for questions
                        }
                    },
                    "required": ["content", "question"]
                }
            }
        }
    ]