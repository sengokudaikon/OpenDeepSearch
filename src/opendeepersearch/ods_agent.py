import asyncio
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Set
from urllib.parse import urlparse

from dotenv import load_dotenv
from litellm import acompletion, utils

from opendeepersearch.context_building.build_context import build_context
from opendeepersearch.context_building.process_sources_pro import SourceProcessor
from opendeepersearch.prompts import (
    ITERATIVE_SEARCH_PROMPT,
    PERPLEXITY_STYLE_PROMPT,
    SEARCH_SYSTEM_PROMPT,
)
from opendeepersearch.serp_search.serp_search import SearchResult, create_search_api

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
            searxng_api_key=searxng_api_key,
        )

        if source_processor_config is None:
            source_processor_config = {}
        if reranker:
            source_processor_config["reranker"] = reranker

        self.source_processor = SourceProcessor(**source_processor_config)

        self.model = (
            model
            if model is not None
            else os.getenv(
                "LITELLM_SEARCH_MODEL_ID",
                os.getenv("LITELLM_MODEL_ID", "openrouter/google/gemini-2.0-flash-001"),
            )
        )
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.iterative_system_prompt = ITERATIVE_SEARCH_PROMPT
        self.perplexity_style_prompt = PERPLEXITY_STYLE_PROMPT

        openai_base_url = os.environ.get("OPENAI_BASE_URL")
        if openai_base_url:
            utils.set_provider_config("openai", {"base_url": openai_base_url})

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

    def _filter_new_sources(self, sources_data: Dict[str, Any], visited_urls: Set[str]) -> Dict[str, Any]:
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
        accumulated_context: Dict[str, Dict[str, Any]],
    ) -> str:
        """Generate an analysis of current findings when no new sources are found."""
        current_summary = self._summarize_context(accumulated_context)

        try:
            analysis_messages = [
                {
                    "role": "system",
                    "content": "You are a research analysis assistant helping determine next steps.",
                },
                {
                    "role": "user",
                    "content": f"Initial Query: {original_query}\nCurrent Query: {current_query}\n\nNo new sources were found for the current query."
                    f" Based on our previous findings, analyze what we know and what gaps remain."
                    f" Suggest whether we should: (1) try a different search query (and what it should be), or (2) proceed to final synthesis with what we have.\n"
                    f"\nPrevious Findings Summary:\n{current_summary}",
                },
            ]

            analysis_response = await acompletion(
                model=self.model, messages=analysis_messages, temperature=0.2, top_p=self.top_p
            )
            return analysis_response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing current findings: {e}")
            return (
                "Error analyzing current findings. Recommend proceeding to final synthesis with available information."
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

    def _extract_next_action(self, analysis_content: str) -> str:
        """Parse the LLM analysis to extract the next search query or action."""

        stop_indicators = [
            "SYNTHESIZE_FINAL",
            "sufficient information gathered",
            "enough information",
            "information is complete",
            "We can now provide a final answer",
        ]

        for indicator in stop_indicators:
            if indicator in analysis_content:
                return "SYNTHESIZE_FINAL"

        query_markers = [
            r"Next search query: [\"']?([^\"'\n]+)[\"']?",
            r"Next query: [\"']?([^\"'\n]+)[\"']?",
            r"For the next iteration, search for: [\"']?([^\"'\n]+)[\"']?",
            r"Follow-up query: [\"']?([^\"'\n]+)[\"']?",
            r"Next, we should search for [\"']?([^\"'\n]+)[\"']?",
        ]

        for marker in query_markers:
            match = re.search(marker, analysis_content)
            if match:
                return match.group(1).strip()

        if any(
            phrase in analysis_content
            for phrase in [
                "need more information",
                "additional research",
                "further details",
                "explore further",
            ]
        ):

            topics_match = re.search(
                r"(need|should|could) (search|research|explore|investigate|find out|learn) (about|on|for|more about) ([^.]+)",
                analysis_content,
            )
            if topics_match:
                return topics_match.group(4).strip()

        return ""

    def _update_accumulated_context(
        self,
        accumulated_context: Dict[str, Dict[str, Any]],
        iteration_number: int,
        processed_data: Dict[str, Any],
        query: str,
        context_string: str,
        analysis: str = None,
    ) -> None:
        """Update the accumulated context with new findings from this iteration."""

        if iteration_number not in accumulated_context:
            accumulated_context[iteration_number] = {}

        if processed_data and "organic" in processed_data:
            sources = processed_data.get("organic", [])
            accumulated_context[iteration_number]["sources"] = sources

        accumulated_context[iteration_number]["context"] = context_string
        accumulated_context[iteration_number]["query"] = query

        if analysis:

            findings_match = re.search(
                r"(?:Key findings|Key insights|Main points|Summary of findings)(.*?)(?:Next steps|Further research|Follow-up|$)",
                analysis,
                re.DOTALL | re.IGNORECASE,
            )

            if findings_match:
                key_findings = findings_match.group(1).strip()
                accumulated_context[iteration_number]["key_findings"] = key_findings
            else:

                accumulated_context[iteration_number]["key_findings"] = analysis

    def _build_final_context(self, accumulated_context: Dict[str, Dict[str, Any]]) -> str:
        """Build a comprehensive context string from all accumulated findings for final synthesis."""
        if not accumulated_context:
            return ""

        context_parts = []

        context_parts.append("# SUMMARY OF RESEARCH\n")
        context_parts.append(self._summarize_context(accumulated_context))

        context_parts.append("\n\n# DETAILED FINDINGS\n")

        for iteration, data in sorted(accumulated_context.items()):
            if "query" in data and "context" in data:
                context_parts.append(f"\n## Iteration {iteration}: '{data['query']}'\n")

                context_content = data["context"]
                if len(context_content) > 2000:
                    context_content = context_content[:2000] + "...\n[Content truncated for brevity]"

                context_parts.append(context_content)

                if "sources" in data and data["sources"]:
                    context_parts.append("\nSources for this iteration:")
                    for i, source in enumerate(data["sources"][:5]):
                        title = source.get("title", "Unknown")
                        link = source.get("link", "#")
                        context_parts.append(f"- [{title}]({link})")

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
        max_sources: int = 3,
        min_sources: int = 1,
        pro_mode: bool = False,
        max_iterations: int = 3,
        max_sources_per_iteration: int = 10,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Searches for information, generates an AI response, yielding intermediate steps and sources.

        This method combines web search, context building, and AI completion. It yields
        dictionaries representing the state of the process (status, sources found, etc.)
        and finally yields the answer and the list of processed sources used.

        Args:
            query (str): The question or query to answer.
            max_sources (int, default=3): Maximum number of sources to include in the context for normal mode.
            min_sources (int, default=1): Minimum number of unique sources to include. If greater
                than 0, forces the model to search for at least this many unique sources.
            pro_mode (bool, default=False): When enabled, performs deep research mode with
                iterative search workflow and transparent reasoning in a conversational style.
            max_iterations (int, default=3): Maximum number of search iterations to perform in deep research mode.
            max_sources_per_iteration (int, default=10): Maximum number of sources to process per iteration in deep research mode.

        Yields:
            Dict[str, Any]: Dictionaries representing intermediate steps or the final result.
                Possible types: "status", "sources_found", "final_answer", "thought", etc.
        """
        if not pro_mode:

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
                max_sources,
                query,
                False,
                min_sources,
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

            current_query = query
            accumulated_context = {}
            visited_urls = set()
            iteration_count = 0

            try:
                planning_prompt = (
                    f"Based on the query '{query}', what would be a good systematic research approach?"
                    f" Consider what initial search terms would be most effective, what types of sources might have reliable information, and what sub-questions we might need to explore."
                    f" Respond in a conversational first-person style as if you're thinking out loud about your research strategy."
                )

                plan_messages = [
                    {"role": "system", "content": self.perplexity_style_prompt},
                    {"role": "user", "content": planning_prompt},
                ]

                plan_response = await acompletion(model=self.model, messages=plan_messages, temperature=0.3, top_p=0.8)

                initial_plan = plan_response.choices[0].message.content
                yield {"type": "thought", "content": initial_plan}
            except Exception as e:
                yield {"type": "warning", "content": f"Could not generate initial plan: {e}"}

            while iteration_count < max_iterations:
                iteration_count += 1

                yield {"type": "thought", "content": f'*Searching*\n"{current_query}"'}

                try:
                    sources_result: SearchResult[Dict[str, Any]] = await asyncio.to_thread(
                        self.serp_search.get_sources, current_query
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
                if not new_sources_data or not new_sources_data.get("organic"):
                    yield {
                        "type": "thought",
                        "content": f"My search for '{current_query}' didn't yield any new relevant sources. I'll need to refine my approach.",
                    }

                    analysis_content = await self._analyze_current_findings(query, current_query, accumulated_context)
                    yield {"type": "thought", "content": analysis_content}

                    next_query_or_action = self._extract_next_action(analysis_content)
                    if next_query_or_action == "SYNTHESIZE_FINAL":
                        break
                    elif next_query_or_action:
                        current_query = next_query_or_action
                        continue
                    else:

                        break

                source_urls = [s for s in new_sources_data.get("organic", []) if s.get("link")]
                visited_urls.update([s.get("link") for s in source_urls if s.get("link")])

                favicons_list = self._format_sources_as_favicons(source_urls)
                yield {"type": "thought", "content": f"*Reading*\n{favicons_list}"}

                processed_iteration_data = {}
                try:
                    async for update in self.source_processor.process_sources(
                        SearchResult(data=new_sources_data),
                        max_sources_per_iteration,
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
                if not iteration_context_string.strip():
                    yield {
                        "type": "thought",
                        "content": "I wasn't able to extract useful content from these sources. Let me try a different approach.",
                    }
                    continue

                try:
                    previous_findings_summary = self._summarize_context(accumulated_context)

                    analysis_prompt = f"""
                    Based on the information I've gathered so far about the query '{query}', I need to analyze what I've just learned.
                    
                    Previous findings summary:
                    {previous_findings_summary}
                    
                    New information from latest search:
                    {iteration_context_string}
                    
                    Please help me:
                    1. Analyze what I've learned
                    2. Identify gaps or inconsistencies
                    3. Determine what follow-up information I need
                    4. Suggest a specific next search query or indicate if I have sufficient information
                    
                    Respond as if you are me thinking out loud in a conversational tone, similar to how Perplexity shows its reasoning.
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

                    self._update_accumulated_context(
                        accumulated_context,
                        iteration_count,
                        processed_iteration_data,
                        current_query,
                        iteration_context_string,
                        analysis_content,
                    )

                    next_query_or_action = self._extract_next_action(analysis_content)

                    if next_query_or_action == "SYNTHESIZE_FINAL":
                        yield {"type": "thought", "content": "*Writing research report*"}
                        break
                    elif next_query_or_action:
                        current_query = next_query_or_action
                    else:

                        try:
                            refine_prompt = f"""
                            Based on my research so far on '{query}', I need to determine what to search for next.
                            
                            What I've learned so far:
                            {self._summarize_context(accumulated_context)}
                            
                            What specific information should I search for next to better answer the original question? 
                            Respond in first person as if you are me deciding what to search for next. 
                            End your response with the exact search query in quotes, e.g., "search query here".
                            """

                            refine_messages = [
                                {"role": "system", "content": self.perplexity_style_prompt},
                                {"role": "user", "content": refine_prompt},
                            ]

                            refine_response = await acompletion(
                                model=self.model,
                                messages=refine_messages,
                                temperature=0.3,
                                top_p=0.8,
                            )

                            refine_content = refine_response.choices[0].message.content

                            query_match = re.search(r'"([^"]+)"', refine_content)
                            if query_match:
                                current_query = query_match.group(1)

                            else:

                                sentences = re.split(r"[.!?]", refine_content)
                                if sentences and len(sentences) > 0:
                                    best_sentence = max(sentences, key=len)
                                    current_query = best_sentence.strip()
                                    if len(current_query) < 10:
                                        current_query = f"more information about {query}"

                        except Exception as e:
                            yield {"type": "warning", "content": f"Error refining query: {e}"}
                            yield {
                                "type": "thought",
                                "content": "I'm having trouble determining what to search for next. I'll try a more general search to continue my research.",
                            }
                            current_query = f"latest information about {query}"

                except Exception as e:
                    yield {
                        "type": "error",
                        "content": f"Error during iteration {iteration_count} analysis: {e}",
                    }

                    continue

            yield {
                "type": "thought",
                "content": "Now I'll synthesize all the information I've gathered into a comprehensive answer.",
            }
            final_context_string = self._build_final_context(accumulated_context)

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
