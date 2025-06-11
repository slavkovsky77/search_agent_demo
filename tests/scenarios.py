"""
Test scenario definitions for the Internet Search Agent
"""
from dataclasses import dataclass


@dataclass
class TestScenario:
    """Test scenario configuration."""
    name: str
    request: str
    expected_count: int
    content_type: str  # "images", "articles", "webpage"
    expected_topic: str | None = None
    expected_source: str | None = None
    url_validation: str | None = None


# Define all test scenarios based on launch.json
TEST_SCENARIOS = [
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
    ),
    TestScenario(
        name="hackernews_frontpage",
        request="Download the front page of https://news.ycombinator.com/",
        expected_count=1,
        content_type="webpage",
        url_validation="news.ycombinator.com"
    )
]
