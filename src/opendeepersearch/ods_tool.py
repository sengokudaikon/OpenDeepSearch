# pylint: disable=line-too-long
# flake8: noqa: E501

from typing import Any, AsyncGenerator, Dict, Literal, Optional

from smolagents import Tool

from opendeepersearch.ods_agent import OpenDeepSearchAgent


class OpenDeepSearchTool(Tool):
    name = "web_search"
    description = """
    Performs web search based on your query (think a Google search) then returns the final answer that is processed by an llm."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform",
        },
        "min_sources": {
            "type": "integer",
            "description": "Minimum number of unique sources to include (default: 3)",
            "default": 3,
            "nullable": True,
        },
        "max_sources": {
            "type": "integer",
            "description": "Maximum number of sources to include (default: 10)",
            "default": 10,
            "nullable": True,
        },
        "pro_mode": {
            "type": "boolean",
            "description": "When true, uses deep research mode with iterative workflow and transparent, conversational reasoning style (default: false)",
            "default": False,
            "nullable": True,
        },
        "max_iterations": {
            "type": "integer",
            "description": "Maximum number of search iterations for deep research mode (default: 3)",
            "default": 3,
            "nullable": True,
        },
        "max_sources_per_iteration": {
            "type": "integer",
            "description": "Maximum sources to process per iteration in deep research mode (default: 10)",
            "default": 10,
            "nullable": True,
        },
    }

    output_type = "string"

    def __init__(
        self,
        model_name: Optional[str] = None,
        reranker: str = "infinity",
        search_provider: Literal["serper", "searxng"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None,
    ):
        super().__init__()
        self.search_model_name = model_name
        self.reranker = reranker
        self.search_provider = search_provider
        self.serper_api_key = serper_api_key
        self.searxng_instance_url = searxng_instance_url
        self.searxng_api_key = searxng_api_key

    async def forward(
        self,
        query: str,
        min_sources: int = 0,
        max_sources: int = 10,
        pro_mode: bool = False,
        max_iterations: int = 3,
        max_sources_per_iteration: int = 10,
    ) -> Dict[str, Any]:
        """
        Standard forward method for smol-agents compatibility.
        Runs the full streaming process and returns only the final answer dictionary.

        Args:
            query (str): The search query to perform
            min_sources (int): Minimum number of unique sources to include
            max_sources (int): Maximum number of sources to include
            pro_mode (bool): When true, uses deep research mode with iterative workflow and transparent reasoning
            max_iterations (int): Maximum number of search iterations for deep research mode
            max_sources_per_iteration (int): Maximum sources to process per iteration in deep research mode
        """
        final_result = {}
        async for result in self.stream_forward(
            query=query,
            min_sources=min_sources,
            max_sources=max_sources,
            pro_mode=pro_mode,
            max_iterations=max_iterations,
            max_sources_per_iteration=max_sources_per_iteration,
        ):
            if result.get("type") == "final_answer":
                final_result = result
                break
            elif result.get("type") == "error":
                final_result = result
                break

        return final_result

    async def stream_forward(
        self,
        query: str,
        min_sources: int = 0,
        max_sources: int = 10,
        pro_mode: bool = False,
        max_iterations: int = 3,
        max_sources_per_iteration: int = 10,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Calls the OpenDeepSearchAgent's ask method and yields the streamed results.
        This is the method to use for getting intermediate updates.

        Args:
            query (str): The search query to perform
            min_sources (int): Minimum number of unique sources to include
            max_sources (int): Maximum number of sources to include
            pro_mode (bool): When true, uses deep research mode with iterative workflow and transparent reasoning
            max_iterations (int): Maximum number of search iterations for deep research mode
            max_sources_per_iteration (int): Maximum sources to process per iteration in deep research mode
        """

        if not hasattr(self, "search_tool") or self.search_tool is None:
            self.setup()

        async for result in self.search_tool.ask(
            query=query,
            max_sources=max_sources,
            min_sources=min_sources,
            pro_mode=pro_mode,
            max_iterations=max_iterations,
            max_sources_per_iteration=max_sources_per_iteration,
        ):
            yield result

    def setup(self):
        self.search_tool = OpenDeepSearchAgent(
            self.search_model_name,
            reranker=self.reranker,
            search_provider=self.search_provider,
            serper_api_key=self.serper_api_key,
            searxng_instance_url=self.searxng_instance_url,
            searxng_api_key=self.searxng_api_key,
        )
