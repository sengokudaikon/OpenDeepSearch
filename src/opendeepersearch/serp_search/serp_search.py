from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar

import requests

from opendeepersearch.config import config

T = TypeVar("T")


class SearchAPIError(Exception):
    """Custom exception for Search API related errors"""

    pass


class SerperAPIError(SearchAPIError):
    """Custom exception for Serper API related errors"""

    pass


class SearXNGError(SearchAPIError):
    """Custom exception for SearXNG related errors"""

    pass


@dataclass
class SerperConfig:
    """Configuration for Serper API"""

    api_key: str
    api_url: str = "https://google.serper.dev/search"
    default_location: str = "us"
    timeout: int = 10

    @classmethod
    def from_env(cls) -> "SerperConfig":
        """Create config from environment variables"""
        api_key = config.serper.api_key
        if not api_key:
            raise SerperAPIError("SERPER_API_KEY not configured in environment or .env file")
        return cls(
            api_key=api_key,
            api_url=str(config.serper.api_url),
            default_location=config.serper.location,
            timeout=config.serper.timeout,
        )


@dataclass
class SearXNGConfig:
    """Configuration for SearXNG instance"""

    instance_url: str
    api_key: Optional[str] = None
    default_location: str = "all"
    timeout: int = 10

    @classmethod
    def from_env(cls) -> "SearXNGConfig":
        """Create config from environment variables"""
        instance_url = config.searxng.instance_url
        if not instance_url:
            raise SearXNGError("SEARXNG_INSTANCE_URL not configured in environment or .env file")
        api_key = config.searxng.api_key
        return cls(
            instance_url=str(instance_url),
            api_key=api_key,
            default_location=config.searxng.location,
            timeout=config.searxng.timeout,
        )


class SearchResult(Generic[T]):
    """Container for search results with error handling"""

    def __init__(self, data: Optional[T] = None, error: Optional[str] = None):
        self.data = data
        self.error = error
        self.success = error is None

    @property
    def failed(self) -> bool:
        return not self.success


class SearchAPI(ABC):
    """Abstract base class for search APIs"""

    @abstractmethod
    def get_sources(
        self, query: str, num_results: int = 8, stored_location: Optional[str] = None
    ) -> SearchResult[Dict[str, Any]]:
        """Get search results from the API"""
        pass


class SerperAPI(SearchAPI):
    def __init__(self, api_key: Optional[str] = None, config_override: Optional[SerperConfig] = None):
        if config_override:
            self.config = config_override
        elif api_key:
            # If only api_key is provided, create config using it and defaults from global config
            self.config = SerperConfig(
                api_key=api_key,
                api_url=str(config.serper.api_url),
                default_location=config.serper.location,
                timeout=config.serper.timeout,
            )
        else:
            # Default: load everything from global config/env
            self.config = SerperConfig.from_env()

        self.headers = {"X-API-KEY": self.config.api_key, "Content-Type": "application/json"}

    @staticmethod
    def extract_fields(items: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
        """Extract specified fields from a list of dictionaries"""
        return [{key: item.get(key, "") for key in fields if key in item} for item in items]

    def get_sources(
        self, query: str, num_results: int = 8, stored_location: Optional[str] = None
    ) -> SearchResult[Dict[str, Any]]:
        """
        Fetch search results from Serper API.

        Args:
            query: Search query string
            num_results: Number of results to return (default: 8, max: 10)
            stored_location: Optional location string

        Returns:
            SearchResult containing the search results or error information
        """
        if not query.strip():
            return SearchResult(error="Query cannot be empty")

        try:
            search_location = (stored_location or self.config.default_location).lower()

            payload = {"q": query, "num": min(max(1, num_results), 10), "gl": search_location}

            response = requests.post(
                self.config.api_url, headers=self.headers, json=payload, timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()

            results = {
                "organic": self.extract_fields(data.get("organic", []), ["title", "link", "snippet", "date"]),
                "topStories": self.extract_fields(data.get("topStories", []), ["title", "imageUrl"]),
                "images": self.extract_fields(data.get("images", [])[:6], ["title", "imageUrl"]),
                "graph": data.get("knowledgeGraph"),
                "answerBox": data.get("answerBox"),
                "peopleAlsoAsk": data.get("peopleAlsoAsk"),
                "relatedSearches": data.get("relatedSearches"),
            }

            return SearchResult(data=results)

        except requests.RequestException as e:
            return SearchResult(error=f"API request failed: {str(e)}")
        except Exception as e:
            return SearchResult(error=f"Unexpected error: {str(e)}")


class SearXNGAPI(SearchAPI):
    """API client for SearXNG search engine"""

    def __init__(
        self,
        instance_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[SearXNGConfig] = None,
    ):
        if instance_url:
            self.config = SearXNGConfig(instance_url=instance_url, api_key=api_key)
        else:
            self.config = config or SearXNGConfig.from_env()

        self.headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            self.headers["X-API-Key"] = self.config.api_key

    def get_sources(
        self, query: str, num_results: int = 8, stored_location: Optional[str] = None
    ) -> SearchResult[Dict[str, Any]]:
        """
        Fetch search results from SearXNG instance.

        Args:
            query: Search query string
            num_results: Number of results to return (default: 8)
            stored_location: Optional location string (may not be supported by all instances)

        Returns:
            SearchResult containing the search results or error information
        """
        if not query.strip():
            return SearchResult(error="Query cannot be empty")

        try:

            search_url = self.config.instance_url
            if not search_url.endswith("/search"):
                search_url = search_url.rstrip("/") + "/search"

            params = {
                "q": query,
                "format": "json",
                "pageno": 1,
                "categories": "general",
                "language": "all",
                "time_range": None,
                "safesearch": 0,
                "engines": "google,bing,duckduckgo",
                "max_results": min(max(1, num_results), 20),
            }

            if stored_location and stored_location != "all":
                params["language"] = stored_location

            response = requests.get(search_url, headers=self.headers, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()

            organic_results = []
            for result in data.get("results", [])[:num_results]:
                organic_results.append(
                    {
                        "title": result.get("title", ""),
                        "link": result.get("url", ""),
                        "snippet": result.get("content", ""),
                        "date": result.get("publishedDate", ""),
                    }
                )

            image_results = []
            for result in data.get("results", []):
                if result.get("img_src"):
                    image_results.append({"title": result.get("title", ""), "imageUrl": result.get("img_src", "")})
            image_results = image_results[:6]

            results = {
                "organic": organic_results,
                "images": image_results,
                "topStories": [],
                "graph": None,
                "answerBox": None,
                "peopleAlsoAsk": None,
                "relatedSearches": data.get("suggestions", []),
            }

            return SearchResult(data=results)

        except requests.RequestException as e:
            return SearchResult(error=f"SearXNG API request failed: {str(e)}")
        except Exception as e:
            return SearchResult(error=f"Unexpected error with SearXNG: {str(e)}")


def create_search_api(
    search_provider: str = "serper",
    serper_api_key: Optional[str] = None,
    searxng_instance_url: Optional[str] = None,
    searxng_api_key: Optional[str] = None,
) -> SearchAPI:
    """
    Factory function to create the appropriate search API client.

    Args:
        search_provider: The search provider to use ('serper' or 'searxng')
        serper_api_key: Optional API key for Serper
        searxng_instance_url: Optional SearXNG instance URL
        searxng_api_key: Optional API key for SearXNG instance

    Returns:
        An instance of a SearchAPI implementation

    Raises:
        ValueError: If an invalid search provider is specified
    """
    if search_provider.lower() == "serper":
        return SerperAPI(api_key=serper_api_key)
    elif search_provider.lower() == "searxng":
        return SearXNGAPI(instance_url=searxng_instance_url, api_key=searxng_api_key)
    else:
        raise ValueError(f"Invalid search provider: {search_provider}. Must be 'serper' or 'searxng'")
