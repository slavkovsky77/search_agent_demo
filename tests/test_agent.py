#!/usr/bin/env python3
"""
Integration tests for the Internet Search Agent
Tests actually run the agent and validate results with LLM
"""

import pytest
import os
from pathlib import Path

from src.agent_search_v2 import InternetSearchAgent
from src.config import setup_logging
from .scenarios import TEST_SCENARIOS
from .validators import ContentValidator

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
def test_downloads_dir(scope="session") -> Path:
    """Create test download directory organized by scenario."""
    test_dir = Path("test_downloads")
    # shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(exist_ok=True)
    return test_dir


class TestScenarios:
    """Test all scenarios from launch.json configurations."""

    @pytest.mark.parametrize("scenario", TEST_SCENARIOS, ids=[s.name for s in TEST_SCENARIOS])
    def test_scenario(self, scenario, agent: InternetSearchAgent, test_downloads_dir: Path) -> None:
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
