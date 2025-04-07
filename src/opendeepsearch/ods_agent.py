from typing import Optional, Dict, Any, Literal
from opendeepsearch.serp_search.serp_search import create_search_api
from opendeepsearch.context_building.process_sources_pro import SourceProcessor
from opendeepsearch.context_building.build_context import build_context
from litellm import completion, utils
from dotenv import load_dotenv
import os
from opendeepsearch.prompts import SEARCH_SYSTEM_PROMPT
import asyncio
import nest_asyncio
load_dotenv()

class OpenDeepSearchAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = SEARCH_SYSTEM_PROMPT,
        search_provider: Literal["serper", "searxng"] = "serper",
        serper_api_key: Optional[str] = None,
        searxng_instance_url: Optional[str] = None,
        searxng_api_key: Optional[str] = None,
        source_processor_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        top_p: float = 0.3,
        reranker: Optional[str] = "None",
    ):
        """
        Initialize an OpenDeepSearch agent that combines web search, content processing, and LLM capabilities.

        This agent performs web searches using either SerperAPI or SearXNG, processes the search results to extract
        relevant information, and uses a language model to generate responses based on the gathered context.

        Args:
            model (str): The identifier for the language model to use (compatible with LiteLLM).
            system_prompt (str, optional): Custom system prompt for the language model. If not provided,
                uses a default prompt that instructs the model to answer based on context.
            search_provider (str, optional): The search provider to use ('serper' or 'searxng'). Default is 'serper'.
            serper_api_key (str, optional): API key for SerperAPI. Required if search_provider is 'serper' and
                SERPER_API_KEY environment variable is not set.
            searxng_instance_url (str, optional): URL of the SearXNG instance. Required if search_provider is 'searxng'
                and SEARXNG_INSTANCE_URL environment variable is not set.
            searxng_api_key (str, optional): API key for SearXNG instance. Optional even if search_provider is 'searxng'.
            source_processor_config (Dict[str, Any], optional): Configuration dictionary for the
                SourceProcessor. Supports the following options:
                - strategies (List[str]): Content extraction strategies to use
                - filter_content (bool): Whether to enable content filtering
                - top_results (int): Number of top results to process
            temperature (float, default=0.2): Controls randomness in model outputs. Lower values make
                the output more focused and deterministic.
            top_p (float, default=0.3): Controls nucleus sampling for model outputs. Lower values make
                the output more focused on high-probability tokens.
            reranker (str, optional): Identifier for the reranker to use. If not provided,
                uses the default reranker from SourceProcessor.
        """
        self.serp_search = create_search_api(
            search_provider=search_provider,
            serper_api_key=serper_api_key,
            searxng_instance_url=searxng_instance_url,
            searxng_api_key=searxng_api_key
        )

        if source_processor_config is None:
            source_processor_config = {}
        if reranker:
            source_processor_config['reranker'] = reranker

        self.source_processor = SourceProcessor(**source_processor_config)

        self.model = model if model is not None else os.getenv("LITELLM_SEARCH_MODEL_ID", os.getenv("LITELLM_MODEL_ID", "openrouter/google/gemini-2.0-flash-001"))
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt

        openai_base_url = os.environ.get("OPENAI_BASE_URL")
        if openai_base_url:
            utils.set_provider_config("openai", {"base_url": openai_base_url})

    async def search_and_build_context(
        self,
        query: str,
        max_sources: int = 2,
        pro_mode: bool = False
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Performs a web search, processes sources, and builds context.

        This method executes a search query, processes the returned sources, and builds a
        consolidated context string, inspired by FreshPrompt in the FreshLLMs paper, that can be used for answering questions. It also returns the list of processed source dictionaries.

        Args:
            query (str): The search query to execute.
            max_sources (int, default=2): Maximum number of sources to process. If pro_mode
                is enabled, this overrides the top_results setting in source_processor_config
                when it's smaller.
            pro_mode (bool, default=False): When enabled, performs a deeper search and more
                thorough content processing.

        Returns:
            tuple[str, list[dict[str, Any]]]: A tuple containing:
                - The formatted context string.
                - A list of dictionaries, each representing a processed source.
        """
        sources = self.serp_search.get_sources(query)

        processed_sources = await self.source_processor.process_sources(
            sources,
            max_sources,
            query,
            pro_mode
        )

        context_string = build_context(processed_sources)

        return context_string, processed_sources

    async def ask(
        self,
        query: str,
        max_sources: int = 2,
        pro_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Searches for information, generates an AI response, and returns sources.

        This method combines web search, context building, and AI completion. It gathers
        information, uses an LLM to generate a response based on the context, and returns
        both the answer and the list of processed sources used.

        Args:
            query (str): The question or query to answer.
            max_sources (int, default=2): Maximum number of sources to include in the context.
            pro_mode (bool, default=False): When enabled, performs a more comprehensive search
                and analysis of sources.

        Returns:
            dict[str, Any]: A dictionary containing:
                - "answer" (str): The AI-generated response.
                - "sources" (list[dict[str, Any]]): The list of processed sources.
        """
        context, sources = await self.search_and_build_context(query, max_sources, pro_mode)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        response = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources
        }

    def ask_sync(
        self,
        query: str,
        max_sources: int = 2,
        pro_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Synchronous version of ask() method. Returns a dictionary with answer and sources.
        """
        try:
            # Try getting the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in a running event loop (e.g., Jupyter), use nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            # If there's no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.ask(query, max_sources, pro_mode))
