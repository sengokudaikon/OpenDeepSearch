import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Set, Tuple
from urllib.parse import urlparse

from litellm import acompletion, utils

from opendeepersearch.config import config
from opendeepersearch.context_building.build_context import build_context
from opendeepersearch.context_building.process_sources_pro import SourceProcessor
from opendeepersearch.prompts import (
    ITERATIVE_SEARCH_PROMPT,
    PERPLEXITY_STYLE_PROMPT,
    PLANNING_PROMPT,
    PRE_FILTERING_PROMPT,
    SEARCH_SYSTEM_PROMPT,
)
from opendeepersearch.serp_search.serp_search import SearchResult, create_search_api

logger = logging.getLogger(__name__)


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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        reranker: Optional[str] = "jina",
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
            temperature (float, default=None): Controls randomness in model outputs. If None, uses the default
                temperature from the configuration.
            top_p (float, default=None): Controls nucleus sampling for model outputs. If None, uses the default
                top_p from the configuration.
            reranker (str, optional): Identifier for the reranker to use. If not provided,
                uses the default reranker from SourceProcessor.
        """
        self.serp_search = create_search_api(
            search_provider=search_provider,
            serper_api_key=serper_api_key,
            searxng_instance_url=searxng_instance_url,
            searxng_api_key=searxng_api_key,
        )

        if source_processor_config is None:
            source_processor_config = {}
        if reranker:
            source_processor_config["reranker"] = reranker

        self.source_processor = SourceProcessor(**source_processor_config)

        self.model = model if model is not None else (config.litellm.search_model_id or config.litellm.model_id)
        self.temperature = temperature if temperature is not None else config.llm_generation.temperature
        self.top_p = top_p if top_p is not None else config.llm_generation.top_p
        self.system_prompt = system_prompt
        self.iterative_system_prompt = ITERATIVE_SEARCH_PROMPT
        self.perplexity_style_prompt = PERPLEXITY_STYLE_PROMPT
        self.orchestrator_model = config.litellm.orchestrator_model_id or self.model
        self.planning_prompt_template = PLANNING_PROMPT

        openai_base_url = config.openai.base_url
        if openai_base_url:
            utils.set_provider_config("openai", {"base_url": str(openai_base_url)})  # type: ignore[attr-defined]

    def _get_domain_from_url(self, url: str) -> str:
        """Extract just the domain name from a URL."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return url

    def _get_favicon_entry(self, url: str) -> str:
        """Format a domain as a favicon entry for perplexity-style output."""
        domain = self._get_domain_from_url(url)
        return f"{domain} favicon\n{domain}"

    def _format_sources_as_favicons(self, sources: List[Dict[str, Any]]) -> str:
        """Format a list of source URLs as favicon entries for perplexity-style output."""
        if not sources:
            return ""

        favicons = []
        for source in sources[:10]:
            url = source.get("link", "")
            if url:
                favicons.append(self._get_favicon_entry(url))

        return "\n".join(favicons)

    def _filter_new_sources(self, sources_data: Optional[Dict[str, Any]], visited_urls: Set[str]) -> Dict[str, Any]:  # type: ignore
        """
        Filter sources to only include URLs that haven't been visited yet.

        Args:
            sources_data (Dict[str, Any]): The search result data containing source URLs
            visited_urls (Set[str]): Set of URLs that have already been processed

        Returns:
            Dict[str, Any]: Filtered search results with only new sources
        """
        if not sources_data:
            return {}

        filtered_data = dict(sources_data)

        if "organic" in filtered_data and filtered_data["organic"] is not None:

            filtered_data["organic"] = [
                source for source in filtered_data["organic"] if source.get("link") not in visited_urls
            ]

        for key in ["knowledgeGraph", "relatedSearches", "peopleAlsoAsk"]:
            if key in filtered_data and filtered_data[key] is not None:
                filtered_data[key] = [
                    item
                    for item in filtered_data[key]
                    if not any(url in visited_urls for url in [item.get("link"), item.get("url")])
                ]

        return filtered_data

    async def _analyze_current_findings(
        self,
        original_query: str,
        current_query: str,
        accumulated_context: Dict[int, Dict[str, Any]],
        iteration_count: int,
        max_iterations: int,
        research_plan: str,
    ) -> str:
        """Generate an analysis when no new sources are found, considering the plan."""
        current_summary = self._summarize_context(accumulated_context)

        try:
            analysis_prompt = f"""
Original User Query: '{original_query}'
Research Plan:
{research_plan}

Current Iteration ({iteration_count}/{max_iterations}) Search Query: '{current_query}'

My search for '{current_query}' did not yield any *new* relevant sources.
I need to decide whether the Research Plan is sufficiently covered or requires further search.

Summary of Previously Gathered Findings:
{current_summary}

Instructions:
1. Compare the 'Summary of Previously Gathered Findings' against each sub-question in the 'Research Plan'.
2. Decide if all plan points are addressed:
   - If yes, conclude with JSON: {{"action": "synthesize"}}
   - If not, identify the most critical unanswered sub-question and conclude with JSON: {{"action": "search", "query": "your specific next search query targeting that sub-question"}}
3. Always end your response with the exact JSON decision object on its own line.

Respond with reasoning, then output the JSON decision.
"""
            analysis_messages = [
                {"role": "system", "content": self.perplexity_style_prompt},
                {"role": "user", "content": analysis_prompt},
            ]

            analysis_response = await acompletion(
                model=self.orchestrator_model,
                messages=analysis_messages,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return analysis_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error analyzing current findings when no new sources found: {e}")
            return (
                "Error during analysis. Based on the lack of new sources, "
                "I will proceed to synthesize with the information gathered so far.\n"
                '{"action": "synthesize"}'
            )

    def _summarize_context(self, accumulated_context: Dict[str, Any]) -> str:
        """Create a concise summary of the accumulated context."""
        if not accumulated_context:
            return "No previous findings."

        summary_parts = []

        for iter_num, iter_data in sorted(accumulated_context.items()):
            if not iter_data.get("key_findings"):
                continue

            summary_parts.append(f"Iteration {iter_num} key findings:")
            summary_parts.append(iter_data["key_findings"])

            if iter_data.get("sources"):
                sources = iter_data["sources"][:3]
                sources_text = "; ".join([f"{s.get('title', 'Unknown')} ({s.get('link', 'no link')})" for s in sources])
                summary_parts.append(f"Top sources: {sources_text}")

        if not summary_parts and any("context" in iter_data for iter_data in accumulated_context.values()):

            latest_iter = max(accumulated_context.keys())
            if accumulated_context[latest_iter].get("context"):
                summary_parts.append("Previously gathered information:")
                context_str = accumulated_context[latest_iter]["context"]

                summary_snippet = context_str[:500]
                if len(context_str) > 500:
                    summary_snippet = summary_snippet[: summary_snippet.rfind(".")] + "."
                summary_parts.append(summary_snippet)

        return "\n\n".join(summary_parts)

    def _update_accumulated_context(
        self,
        accumulated_context: Dict[int, Dict[str, Any]],
        iteration_number: int,
        processed_data: Dict[str, Any],
        query: str,
        context_string: str,
        analysis: Optional[str] = None,
        conflicts: Optional[List[str]] = None,
    ) -> None:
        """Update the accumulated context with new findings from this iteration."""

        if iteration_number not in accumulated_context:
            accumulated_context[iteration_number] = {}

        if processed_data and "organic" in processed_data:
            sources = processed_data.get("organic", [])
            accumulated_context[iteration_number]["sources"] = sources

        accumulated_context[iteration_number]["context"] = context_string
        accumulated_context[iteration_number]["query"] = query
        accumulated_context[iteration_number]["conflicts"] = conflicts or []

        if analysis:
            json_start = analysis.rfind('{"action"')
            conversational = analysis[:json_start].strip() if json_start != -1 else analysis.strip()
            accumulated_context[iteration_number]["key_findings"] = (
                conversational if conversational else "Analysis resulted in decision only."
            )

    def _build_final_context(
        self, accumulated_context: Dict[int, Dict[str, Any]], research_plan: Optional[str] = None
    ) -> str:
        """Build a comprehensive context string including conflicts for final synthesis."""
        if not accumulated_context:
            return ""

        context_parts: List[str] = []
        all_conflicts: List[str] = []

        for data in accumulated_context.values():
            if conflicts := data.get("conflicts"):
                all_conflicts.extend(conflicts)

        if research_plan:
            context_parts.append("# INITIAL RESEARCH PLAN")
            context_parts.append(research_plan)
            context_parts.append("")

        context_parts.append("# SUMMARY OF RESEARCH FINDINGS")
        context_parts.append(self._summarize_context(accumulated_context))
        if all_conflicts:
            unique = sorted(set(all_conflicts))
            context_parts.append("# POTENTIAL CONFLICTS / DISCREPANCIES NOTED")
            for c in unique:
                context_parts.append(f"- {c}")
            context_parts.append("")
        context_parts.append("# DETAILED FINDINGS BY ITERATION")
        for i, data in sorted(accumulated_context.items()):
            context_parts.append(f"## Iteration {i}: '{data.get('query','')}'")
            if content := data.get("context"):
                snippet = content if len(content) <= 1500 else content[:1500] + "... [truncated]"
                context_parts.append(snippet)
            if sources := data.get("sources"):
                context_parts.append("Sources:")
                for s in sources[:3]:
                    if isinstance(s, dict):
                        context_parts.append(f"- [{s.get('title','')}]({s.get('link','#')})")
        return "\n".join(context_parts)

    def _extract_final_sources(self, accumulated_context: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and deduplicate sources from all iterations for the final answer."""
        all_sources = []
        seen_urls = set()

        for iteration_data in accumulated_context.values():
            if "sources" in iteration_data:
                for source in iteration_data["sources"]:
                    url = source.get("link")

                    if url and url not in seen_urls:
                        all_sources.append(source)
                        seen_urls.add(url)

        return all_sources

    async def ask(
        self,
        query: str,
        max_sources: Optional[int] = None,
        min_sources: Optional[int] = None,
        pro_mode: Optional[bool] = None,
        max_iterations: Optional[int] = None,
        max_sources_per_iteration: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Searches for information, generates an AI response, yielding intermediate steps and sources.

        This method combines web search, context building, and AI completion. It yields
        dictionaries representing the state of the process (status, sources found, etc.)
        and finally yields the answer and the list of processed sources used.

        Args:
            query (str): The question or query to answer.
            max_sources (int, optional): Maximum number of sources to include in the context for normal mode.
            min_sources (int, optional): Minimum number of unique sources to include. If greater
                than 0, forces the model to search for at least this many unique sources.
            pro_mode (bool, optional): When enabled, performs deep research mode with
                iterative search workflow and transparent reasoning in a conversational style.
            max_iterations (int, optional): Maximum number of search iterations to perform in deep research mode.
            max_sources_per_iteration (int, optional): Maximum number of sources to process per iteration in deep research mode.

        Yields:
            Dict[str, Any]: Dictionaries representing intermediate steps or the final result.
                Possible types: "status", "sources_found", "final_answer", "thought", etc.
        """
        effective_min_sources = min_sources if min_sources is not None else config.search.min_sources
        effective_max_sources = max_sources if max_sources is not None else config.search.max_sources
        effective_pro_mode = pro_mode if pro_mode is not None else config.search.pro_mode
        effective_max_iterations = max_iterations if max_iterations is not None else config.search.max_iterations
        effective_max_sources_per_iteration = (
            max_sources_per_iteration
            if max_sources_per_iteration is not None
            else config.search.max_sources_per_iteration
        )

        if effective_max_sources < effective_min_sources:
            raise ValueError(
                f"max_sources ({effective_max_sources}) cannot be less than min_sources ({effective_min_sources})"
            )

        logger.info(
            f"Effective search settings: min_sources={effective_min_sources}, "
            f"max_sources={effective_max_sources}, pro_mode={effective_pro_mode}"
        )

        if not effective_pro_mode:

            yield {"type": "status", "content": "Starting research (standard mode)..."}

            try:
                plan_messages = [
                    {"role": "system", "content": "You are a research planning assistant."},
                    {
                        "role": "user",
                        "content": f"Based on the query '{query}', briefly outline the steps you will take to find the answer. Focus on search strategy and information synthesis.",
                    },
                ]
                plan_response = await acompletion(
                    model=self.model, messages=plan_messages, temperature=0.1, top_p=self.top_p
                )
                yield {"type": "thought", "content": plan_response.choices[0].message.content}
            except Exception as e:
                print(f"Error generating initial plan: {e}")
                yield {"type": "warning", "content": "Could not generate initial plan."}

            try:
                initial_sources: SearchResult[Dict[str, Any]] = await asyncio.to_thread(
                    self.serp_search.get_sources, query
                )
            except Exception as e:
                print(f"Error running get_sources in thread: {e}")
                yield {"type": "error", "content": f"Failed to perform search: {e}"}
                return

            source_urls = []
            if initial_sources.failed:
                yield {"type": "warning", "content": f"Search failed: {initial_sources.error}"}
            elif (
                initial_sources.data
                and "organic" in initial_sources.data
                and isinstance(initial_sources.data["organic"], list)
            ):
                source_urls = [s.get("link") for s in initial_sources.data["organic"] if s.get("link")]
            yield {"type": "sources_found", "content": source_urls}

            processed_sources_data: Dict[str, Any] = initial_sources.data if initial_sources.data else {}
            processed_sources_list: List[Dict[str, Any]] = []

            async for update in self.source_processor.process_sources(
                initial_sources,
                effective_max_sources,
                query,
                False,
                effective_min_sources,
            ):
                if update.get("type") == "processed_sources":
                    processed_sources_data = update.get("data", {})
                    if (
                        processed_sources_data
                        and "organic" in processed_sources_data
                        and isinstance(processed_sources_data["organic"], list)
                    ):
                        processed_sources_list = processed_sources_data["organic"]
                else:
                    yield update

            context_string = build_context(processed_sources_data)

            if context_string:
                try:
                    synthesis_messages = [
                        {"role": "system", "content": "You are a research synthesis assistant."},
                        {
                            "role": "user",
                            "content": f"Based on the following context relevant to the query '{query}', briefly summarize the key findings and structure before formulating the final answer:\n"
                            f"\nContext:\n{context_string}",
                        },
                    ]
                    synthesis_response = await acompletion(
                        model=self.model,
                        messages=synthesis_messages,
                        temperature=0.1,
                        top_p=self.top_p,
                    )
                    yield {
                        "type": "thought",
                        "content": synthesis_response.choices[0].message.content,
                    }
                except Exception as e:
                    print(f"Error generating synthesis thought: {e}")
                    yield {"type": "warning", "content": "Could not generate synthesis thought."}
            else:
                yield {"type": "status", "content": "No context built, skipping synthesis thought."}

            yield {"type": "status", "content": "Generating final answer..."}
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context_string}\n\nQuestion: {query}"},
            ]

            try:
                response = await acompletion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                final_answer = response.choices[0].message.content
            except Exception as e:
                print(f"Error during LLM completion: {e}")
                yield {"type": "error", "content": f"Error during LLM completion: {e}"}
                return

            yield {
                "type": "final_answer",
                "answer": final_answer,
                "sources": processed_sources_list,
            }

        else:
            yield {
                "type": "thought",
                "content": "I'll start my research on this topic with a systematic approach, searching for information, analyzing sources, and iteratively refining my search based on what I learn.",
            }
            yield {"type": "status", "content": "Generating initial research plan..."}
            research_plan = ""
            try:
                plan_prompt = self.planning_prompt_template.format(user_query=query)
                plan_resp = await acompletion(
                    model=self.orchestrator_model,
                    messages=[{"role": "user", "content": plan_prompt}],
                    temperature=0.1,
                    top_p=0.5,
                )
                plan_text = plan_resp.choices[0].message.content
                match = re.search(r"Research Plan:\s*\n(.*)$", plan_text, re.DOTALL)
                research_plan = match.group(1).strip() if match else plan_text.strip()
                yield {"type": "research_plan", "content": research_plan}
            except Exception as e:
                logger.warning(f"Could not generate research plan: {e}")
                yield {"type": "warning", "content": f"Proceeding without plan: {e}"}
                research_plan = ""

            original_query = query
            current_query = query
            accumulated_context: Dict[int, Dict[str, Any]] = {}
            visited_urls: Set[str] = set()
            iteration_count = 0

            while True:
                iteration_count += 1
                if iteration_count > effective_max_iterations:
                    yield {
                        "type": "thought",
                        "content": f"Reached maximum iteration limit ({effective_max_iterations}). Proceeding to synthesis.",
                    }
                    yield {"type": "status", "content": f"Max iterations ({effective_max_iterations}) reached."}
                    break

                yield {"type": "thought", "content": f'*Searching*\n"{current_query}"'}

                try:
                    sources_result: SearchResult[Dict[str, Any]] = await asyncio.to_thread(
                        self.serp_search.get_sources,
                        current_query,
                        config.search.num_serp_results_fetch,
                    )
                except Exception as e:
                    yield {
                        "type": "error",
                        "content": f"Failed to perform search in iteration {iteration_count}: {e}",
                    }
                    break

                if sources_result.failed:
                    yield {
                        "type": "warning",
                        "content": f"Search failed in iteration {iteration_count}: {sources_result.error}",
                    }
                    continue

                new_sources_data = self._filter_new_sources(sources_result.data, visited_urls)
                unvisited_results = new_sources_data.get("organic", []) or []
                yield {"type": "status", "content": f"Evaluating {len(unvisited_results)} potential sources..."}
                formatted_results = "\n".join(
                    [
                        f"Title: {s.get('title','')}\nURL: {s.get('link','')}\nSnippet: {s.get('snippet','')}"
                        for s in unvisited_results
                    ]
                )
                pre_filter_prompt = PRE_FILTERING_PROMPT.format(
                    query=original_query,
                    num_sources=config.search.num_sources_pre_filter,
                    results=formatted_results,
                )
                try:
                    pf_response = await acompletion(
                        model=config.search.pre_filtering_model_id or self.orchestrator_model,
                        messages=[{"role": "user", "content": pre_filter_prompt}],
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    match = re.search(r"(\[.*\])", pf_response.choices[0].message.content, re.DOTALL)
                    if match:
                        selected_urls = json.loads(match.group(1))
                    else:
                        raise ValueError("No JSON list found in pre-filter response")
                except Exception as e:
                    logger.warning(f"Pre-filtering failed: {e}")
                    selected_urls = [s.get("link") for s in unvisited_results[: config.search.num_sources_pre_filter]]
                filtered_sources_for_scraping = [s for s in unvisited_results if s.get("link") in selected_urls]
                if not filtered_sources_for_scraping:
                    filtered_sources_for_scraping = unvisited_results[: config.search.num_sources_pre_filter]
                # Enforce max sources per iteration
                filtered_sources_for_scraping = filtered_sources_for_scraping[:effective_max_sources_per_iteration]
                yield {
                    "type": "status",
                    "content": f"Selected {len(filtered_sources_for_scraping)} sources for scraping (max: {effective_max_sources_per_iteration}).",
                }
                visited_urls.update([s.get("link") for s in filtered_sources_for_scraping if s.get("link")])
                filtered_sources_data = {"organic": filtered_sources_for_scraping}
                favicons_list = self._format_sources_as_favicons(filtered_sources_for_scraping)
                yield {"type": "thought", "content": f"*Reading*\n{favicons_list}"}
                processed_iteration_data = {}
                try:
                    async for update in self.source_processor.process_sources(
                        SearchResult(data=filtered_sources_data),
                        len(filtered_sources_for_scraping),
                        current_query,
                        True,
                        min_sources=1,
                    ):
                        if update.get("type") == "processed_sources":
                            processed_iteration_data = update.get("data", {})

                except Exception as e:
                    yield {
                        "type": "warning",
                        "content": f"Error processing sources in iteration {iteration_count}: {e}",
                    }
                    continue

                iteration_context_string = build_context(processed_iteration_data)
                if not iteration_context_string or not iteration_context_string.strip():
                    yield {
                        "type": "thought",
                        "content": "I wasn't able to extract useful content from these sources. Let me try a different approach.",
                    }
                    continue

                try:
                    previous_findings_summary = self._summarize_context(accumulated_context)

                    analysis_prompt = f"""
                    Original User Query: '{original_query}'
                    Current Iteration ({iteration_count}/{effective_max_iterations}) Search Query: '{current_query}'

                    My research goal is to comprehensively answer the Original User Query.
                    I need to analyze the latest findings and decide the next step.

                    Summary of Previous Findings:
                    {previous_findings_summary}

                    New Information Found in This Iteration:
                    {iteration_context_string}

                    Instructions:
                    1. Analyze the 'New Information' in the context of the 'Original User Query'. How does it contribute?
                    2. Evaluate the 'Summary of Previous Findings' combined with the 'New Information'. Is the accumulated knowledge sufficient to provide a complete and accurate answer to the 'Original User Query'?
                    3. Identify the most critical remaining knowledge gaps or inconsistencies preventing a full answer.
                    4. **Decision Time:**
                       - If the information IS sufficient, conclude *only* with: SYNTHESIZE_FINAL
                       - If NOT sufficient, propose a *specific* next search query. Conclude *only* with: Next Query: "your query"
                       - If struggling after several attempts, you may conclude *only* with: SYNTHESIZE_FINAL

                    Respond conversationally with your analysis.
                    CRITICAL: After conversational analysis, you MUST provide the final decision JSON object (`{{"action": "search", "query": "."}}` or `{{"action": "synthesize"}}`) as the *absolute final part* of your response, on its own line(s), with no preceding or succeeding text on that line.
                    """

                    analysis_messages = [
                        {"role": "system", "content": self.perplexity_style_prompt},
                        {"role": "user", "content": analysis_prompt},
                    ]
                    analysis_response = await acompletion(
                        model=self.model, messages=analysis_messages, temperature=0.3, top_p=0.8
                    )
                    analysis_content = analysis_response.choices[0].message.content
                    yield {"type": "thought", "content": analysis_content}

                    logger.debug(f"Full analysis content received:\n{analysis_content}")
                    action, next_query, conflicts = self._parse_action_json(analysis_content)
                    self._update_accumulated_context(
                        accumulated_context,
                        iteration_count,
                        processed_iteration_data,
                        current_query,
                        iteration_context_string,
                        analysis_content,
                        conflicts,
                    )
                    logger.debug(f"Parsed action: {action}, Parsed query: {next_query}, Parsed conflicts: {conflicts}")

                    if action == "synthesize":
                        yield {
                            "type": "thought",
                            "content": "Analysis indicates research is complete. Proceeding to synthesis.",
                        }
                        break
                    elif action == "search" and next_query:
                        if next_query.strip().lower() != current_query.strip().lower():
                            current_query = next_query
                        else:
                            yield {
                                "type": "warning",
                                "content": "Proposed next query is the same as the current one. Breaking loop to synthesize.",
                            }
                            break
                    else:
                        yield {
                            "type": "warning",
                            "content": "Could not determine next action from analysis. Proceeding to synthesis.",
                        }
                        break

                except Exception as e:
                    yield {"type": "error", "content": f"Error during iteration {iteration_count} analysis: {e}"}
                    yield {
                        "type": "thought",
                        "content": "Error during analysis, proceeding to synthesis with available data.",
                    }
                    break

            yield {
                "type": "thought",
                "content": "Now I'll synthesize all the information I've gathered into a comprehensive answer.",
            }
            final_context_string = self._build_final_context(accumulated_context, research_plan)

            if not final_context_string.strip():
                yield {
                    "type": "final_answer",
                    "answer": "I couldn't gather sufficient information to answer the query.",
                    "sources": [],
                }
                return

            final_prompt = f"""
            Based on my research on the original query: '{query}', I've gathered the following information:
            
            {final_context_string}
            
            Please help me synthesize this information into a comprehensive, well-structured report that answers the original query.
            Include relevant citations and organize the information logically.
            """

            final_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": final_prompt},
            ]

            try:
                final_response = await acompletion(
                    model=self.model,
                    messages=final_messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                final_answer = final_response.choices[0].message.content
                final_sources = self._extract_final_sources(accumulated_context)
                yield {"type": "final_answer", "answer": final_answer, "sources": final_sources}
            except Exception as e:
                yield {"type": "error", "content": f"Error during final synthesis: {e}"}

    def _parse_action_json(self, analysis_content: str) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Attempts to parse the final JSON action block from the LLM analysis.

        Args:
            analysis_content: The full text response from the LLM.

        Returns:
            A tuple (action, query, conflicts). 'action' is "search" or "synthesize"; 'query' is next search query or None; 'conflicts' is a list of identified conflicts.
        """
        try:
            json_match = re.search(r'{\s*"action".*?}\s*$', analysis_content, re.DOTALL | re.MULTILINE)
            if not json_match:
                logger.debug("No JSON action block found.")
                return None, None, []
            json_str = json_match.group(0).strip()
            logger.debug(f"Attempting to parse JSON action: {json_str}")
            data = json.loads(json_str)
            action = data.get("action")
            conflicts = data.get("conflicts_found", [])
            if not isinstance(conflicts, list) or not all(isinstance(c, str) for c in conflicts):
                logger.warning(f"Invalid conflicts format: {conflicts}")
                conflicts = []
            if action == "search":
                return "search", data.get("query"), conflicts
            if action == "synthesize":
                return "synthesize", None, conflicts
            logger.warning(f"Unknown action type: {action}")
            return None, None, conflicts
        except json.JSONDecodeError as e:
            logger.warning(f"Failed parsing JSON action: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing: {e}")
        return None, None, []
