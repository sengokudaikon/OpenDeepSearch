import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opendeepersearch.config import config
from opendeepersearch.ods_agent import OpenDeepSearchAgent
from opendeepersearch.ranking_models.infinity_rerank import InfinitySemanticSearcher
from opendeepersearch.serp_search.serp_search import SerperAPI, SerperConfig


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch):
    """Ensure each test gets a fresh config potentially modified by monkeypatch."""
    # Prevent tests from modifying the global config instance directly across tests
    # Instead, tests can modify specific values via monkeypatching AppConfig or its sub-models
    pass


def test_serper_api_init_with_key_uses_global_config_defaults(monkeypatch):
    """Test SerperAPI uses global config URL/location/timeout when only key is provided."""
    # Arrange: Set specific global config values
    test_url = "http://test-serper.com"
    test_location = "gb"
    test_timeout = 5
    monkeypatch.setattr(config.serper, "api_url", test_url)
    monkeypatch.setattr(config.serper, "location", test_location)
    monkeypatch.setattr(config.serper, "timeout", test_timeout)

    # Act: Instantiate with only API key
    api = SerperAPI(api_key="dummy_key")

    # Assert: Check if the instance's config matches the patched global defaults
    assert api.config.api_key == "dummy_key"
    assert api.config.api_url == test_url
    assert api.config.default_location == test_location
    assert api.config.timeout == test_timeout


def test_serper_api_init_with_config_override(monkeypatch):
    """Test SerperAPI uses the provided config override."""
    # Arrange: Create a specific config override
    override_config = SerperConfig(
        api_key="override_key", api_url="http://override-serper.com", default_location="fr", timeout=15
    )
    # Patch global config to ensure override is used, not global
    monkeypatch.setattr(config.serper, "api_url", "http://global-should-not-be-used.com")

    # Act: Instantiate with config_override
    api = SerperAPI(config_override=override_config)

    # Assert: Check if the instance's config matches the override
    assert api.config.api_key == "override_key"
    assert api.config.api_url == "http://override-serper.com"
    assert api.config.default_location == "fr"
    assert api.config.timeout == 15


def test_serper_api_init_default_uses_from_env(monkeypatch):
    """Test SerperAPI uses from_env (global config) by default."""
    # Arrange: Set specific global config values and ensure API key env var is set
    test_url = "http://env-serper.com"
    test_location = "de"
    test_timeout = 8
    monkeypatch.setenv("SERPER_API_KEY", "env_key")  # Required for from_env
    monkeypatch.setattr(config.serper, "api_key", "env_key")  # Ensure config object reflects env
    monkeypatch.setattr(config.serper, "api_url", test_url)
    monkeypatch.setattr(config.serper, "location", test_location)
    monkeypatch.setattr(config.serper, "timeout", test_timeout)

    # Act: Instantiate with default parameters
    api = SerperAPI()

    # Assert: Check if the instance's config matches the patched global defaults via from_env
    assert api.config.api_key == "env_key"
    assert api.config.api_url == test_url
    assert api.config.default_location == test_location
    assert api.config.timeout == test_timeout


def test_infinity_reranker_init_default_uses_global_config(monkeypatch):
    """Test InfinitySemanticSearcher uses global config endpoint by default."""
    # Arrange: Set specific global config value
    test_endpoint = "http://test-infinity:1234/embed"
    monkeypatch.setattr(config.infinity, "endpoint", test_endpoint)

    # Act: Instantiate with default parameters
    reranker = InfinitySemanticSearcher()

    # Assert: Check if the instance uses the patched global config endpoint
    assert reranker.embedding_endpoint == test_endpoint


def test_infinity_reranker_init_with_override(monkeypatch):
    """Test InfinitySemanticSearcher uses the provided endpoint override."""
    # Arrange: Provide an explicit endpoint
    override_endpoint = "http://override-infinity:5678/embed"
    # Patch global config to ensure override is used, not global
    monkeypatch.setattr(config.infinity, "endpoint", "http://global-should-not-be-used.com")

    # Act: Instantiate with explicit endpoint
    reranker = InfinitySemanticSearcher(embedding_endpoint=override_endpoint)

    # Assert: Check if the instance uses the override endpoint
    assert reranker.embedding_endpoint == override_endpoint


@patch("opendeepersearch.ods_agent.create_search_api")  # Mock search API creation
def test_ods_agent_init_default_temperature(mock_create_search, monkeypatch):
    """Test OpenDeepSearchAgent uses global config temperature by default."""
    # Arrange: Set specific global config value
    test_temp = 0.88
    monkeypatch.setattr(config.llm_generation, "temperature", test_temp)
    # Mock necessary env vars if create_search_api relies on them implicitly
    monkeypatch.setenv("SERPER_API_KEY", "dummy_key_for_agent_init")
    monkeypatch.setattr(config.serper, "api_key", "dummy_key_for_agent_init")

    # Act: Instantiate with default parameters
    agent = OpenDeepSearchAgent()

    # Assert: Check if the instance uses the patched global config temperature
    assert agent.temperature == test_temp
    # Ensure default top_p is also loaded
    assert agent.top_p == config.llm_generation.top_p


@patch("opendeepersearch.ods_agent.create_search_api")  # Mock search API creation
def test_ods_agent_init_override_temperature(mock_create_search, monkeypatch):
    """Test OpenDeepSearchAgent uses the provided temperature override."""
    # Arrange: Provide an explicit temperature
    override_temp = 0.11
    # Patch global config to ensure override is used, not global
    monkeypatch.setattr(config.llm_generation, "temperature", 0.99)
    # Mock necessary env vars
    monkeypatch.setenv("SERPER_API_KEY", "dummy_key_for_agent_init")
    monkeypatch.setattr(config.serper, "api_key", "dummy_key_for_agent_init")

    # Act: Instantiate with explicit temperature
    agent = OpenDeepSearchAgent(temperature=override_temp)

    # Assert: Check if the instance uses the override temperature
    assert agent.temperature == override_temp
    # Ensure default top_p is still loaded from global config
    assert agent.top_p == config.llm_generation.top_p


@patch("opendeepersearch.ods_agent.create_search_api")  # Mock search API creation
def test_ods_agent_init_model_default(mock_create_search, monkeypatch):
    """Test OpenDeepSearchAgent uses global config model by default."""
    # Arrange: Set specific global config values
    test_search_model = "test/search-model"
    test_default_model = "test/default-model"
    monkeypatch.setattr(config.litellm, "search_model_id", test_search_model)
    monkeypatch.setattr(config.litellm, "model_id", test_default_model)
    monkeypatch.setenv("SERPER_API_KEY", "dummy_key_for_agent_init")
    monkeypatch.setattr(config.serper, "api_key", "dummy_key_for_agent_init")

    # Act: Instantiate with default model
    agent = OpenDeepSearchAgent()

    # Assert: Check if it prioritizes search_model_id
    assert agent.model == test_search_model


@patch("opendeepersearch.ods_agent.create_search_api")  # Mock search API creation
def test_ods_agent_init_model_fallback(mock_create_search, monkeypatch):
    """Test OpenDeepSearchAgent falls back to model_id if search_model_id is None."""
    # Arrange: Set specific global config values
    test_default_model = "test/default-model"
    monkeypatch.setattr(config.litellm, "search_model_id", None)  # Explicitly None
    monkeypatch.setattr(config.litellm, "model_id", test_default_model)
    monkeypatch.setenv("SERPER_API_KEY", "dummy_key_for_agent_init")
    monkeypatch.setattr(config.serper, "api_key", "dummy_key_for_agent_init")

    # Act: Instantiate with default model
    agent = OpenDeepSearchAgent()

    # Assert: Check if it falls back to model_id
    assert agent.model == test_default_model


@patch("opendeepersearch.ods_agent.create_search_api")  # Mock search API creation
def test_ods_agent_init_model_override(mock_create_search, monkeypatch):
    """Test OpenDeepSearchAgent uses the provided model override."""
    # Arrange: Provide an explicit model
    override_model = "override/model"
    # Patch global config to ensure override is used
    monkeypatch.setattr(config.litellm, "search_model_id", "global/search")
    monkeypatch.setattr(config.litellm, "model_id", "global/default")
    monkeypatch.setenv("SERPER_API_KEY", "dummy_key_for_agent_init")
    monkeypatch.setattr(config.serper, "api_key", "dummy_key_for_agent_init")

    # Act: Instantiate with explicit model
    agent = OpenDeepSearchAgent(model=override_model)

    # Assert: Check if the instance uses the override model
    assert agent.model == override_model
