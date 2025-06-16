#!/usr/bin/env python3
"""
Integration tests for the Internet Search Agent (Function Calling)
Uses existing scenarios and validators but with the function calling agent
"""

import pytest
import os
from pathlib import Path

from src.agent_search import InternetSearchAgent
from src.config import setup_logging
from .scenarios import TEST_SCENARIOS
from .validators import ContentValidator

logger = setup_logging(__name__)


@pytest.fixture
def agent() -> InternetSearchAgent:
    """Create agent instance for testing."""
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
    """Test all scenarios with function calling agent."""

    @pytest.mark.parametrize("scenario", TEST_SCENARIOS, ids=[s.name for s in TEST_SCENARIOS])
    def test_scenario(self, scenario, agent: InternetSearchAgent, test_downloads_dir: Path) -> None:
        """Test a specific scenario with function calling agent."""
        agent.download_dir = test_downloads_dir
        validator = ContentValidator(agent)

        logger.info(f"🎯 Testing scenario: {scenario.name}")
        logger.info(f"📝 Request: {scenario.request}")

        # Execute the request with function calling
        results = agent.execute_request(scenario.request)

        # Validate results (same validation logic as before)
        validator.validate_downloads(results, scenario)

        logger.info(f"🎉 Scenario {scenario.name} completed successfully with function calling!")


class TestFunctionCallingAdvantages:
    """Test specific advantages of function calling over JSON parsing."""

    def test_complex_requests(self, agent: InternetSearchAgent, test_downloads_dir: Path):
        """Test complex requests that might break JSON parsing."""
        agent.download_dir = test_downloads_dir

        complex_requests = [
            "Find 2 photos of cats, please make sure they're cute!",
            "I would like to download 3 articles about AI from any source",
            "Get me 2 photos of mountains (preferably snow-capped ones)",
            "Download 1 article about space exploration from NASA's website",
        ]

        for request in complex_requests:
            logger.info(f"🔧 Testing complex request: {request}")
            results = agent.execute_request(request)

            # Should not fail (function calling is more robust)
            assert isinstance(results, list), f"Should return list, got {type(results)}"
            logger.info(f"✅ Complex request handled successfully: {len(results)} results")

    def test_malformed_requests(self, agent: InternetSearchAgent, test_downloads_dir: Path):
        """Test requests that would break JSON parsing."""
        agent.download_dir = test_downloads_dir

        malformed_requests = [
            "photos... hmm... 2 cats please!",
            "I want (need?) 3 articles about technology",
            "Download \"2\" articles about AI",  # Mixed quotes
            "Get me some photos: 3 elephants would be great!",
        ]

        for request in malformed_requests:
            logger.info(f"🔧 Testing malformed request: {request}")
            results = agent.execute_request(request)

            # Should gracefully handle malformed requests
            assert isinstance(results, list), "Should return list even for malformed request"
            logger.info(f"✅ Malformed request handled gracefully: {len(results)} results")

    def test_edge_cases(self, agent: InternetSearchAgent, test_downloads_dir: Path):
        """Test edge cases that function calling handles better."""
        agent.download_dir = test_downloads_dir

        edge_cases = [
            "Find zero photos of cats",    # Should be rejected by schema (minimum: 1)
            "Find 100 photos of cats",     # Should be capped by schema (maximum: 20)
            "Download articles",           # Missing count, should fail gracefully
            "",                            # Empty request
        ]

        for request in edge_cases:
            logger.info(f"🔧 Testing edge case: '{request}'")
            results = agent.execute_request(request)

            # Should handle edge cases gracefully
            assert isinstance(results, list), "Should return list for edge case"
            logger.info(f"✅ Edge case handled: {len(results)} results")


class TestComparison:
    """Compare function calling vs JSON parsing behavior on the same requests."""

    def test_reliability_comparison(self, agent: InternetSearchAgent, test_downloads_dir: Path):
        """Test that function calling is more reliable than JSON parsing would be."""
        agent.download_dir = test_downloads_dir

        # These requests would potentially cause JSON parsing issues
        potentially_problematic = [
            "Find 3 photos of zebras",           # Good baseline
            "I need 2 articles about science",  # Natural language
            "Get 3 photos of mountains, please",  # Politeness
            "Download 2 articles from NASA",    # Source specification
        ]

        success_count = 0
        for request in potentially_problematic:
            logger.info(f"🔧 Testing reliability: {request}")
            try:
                results = agent.execute_request(request)
                if isinstance(results, list):
                    success_count += 1
                    logger.info(f"✅ Success: {len(results)} results")
                else:
                    logger.warning(f"❌ Unexpected result type: {type(results)}")
            except Exception as e:
                logger.error(f"❌ Failed: {e}")

        # Function calling should handle all requests successfully
        success_rate = success_count / len(potentially_problematic)
        logger.info(f"📊 Function calling success rate: {success_rate:.1%}")

        # Expect high success rate (should be 100% or close)
        assert success_rate >= 0.75, f"Expected high success rate, got {success_rate:.1%}"
