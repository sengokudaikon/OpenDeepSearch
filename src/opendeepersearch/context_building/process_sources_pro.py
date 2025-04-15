import asyncio
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Tuple

from opendeepersearch.context_scraping.crawl4ai_scraper import WebScraper
from opendeepersearch.ranking_models.chunker import Chunker
from opendeepersearch.ranking_models.infinity_rerank import InfinitySemanticSearcher
from opendeepersearch.ranking_models.jina_reranker import JinaReranker
from opendeepersearch.serp_search.serp_search import SearchResult


@dataclass
class Source:
    link: str
    html: str = ""


def _get_valid_sources(sources: SearchResult[Dict[str, Any]], num_elements: int) -> List[Tuple[int, dict]]:
    """
    Get valid sources from search results.

    Args:
        sources: The search results containing source data
        num_elements: Only used to log how many sources were found

    Returns:
        A list of tuples containing the index and source dictionary for valid sources
    """

    if sources.data and "organic" in sources.data:
        organic_results = sources.data["organic"]
        if isinstance(organic_results, list):

            return [(i, source) for i, source in enumerate(organic_results) if source]
    print("Warning: Could not extract valid sources from the search results")
    return []


class SourceProcessor:
    def __init__(
        self,
        top_results: int = 5,
        strategies: List[str] = ["no_extraction"],
        filter_content: bool = True,
        reranker: str = "jina",
    ):
        self.strategies = strategies
        self.filter_content = filter_content
        self.scraper = WebScraper(strategies=self.strategies, filter_content=self.filter_content)
        self.top_results = top_results
        self.chunker = Chunker()

        if reranker.lower() == "jina":
            self.semantic_searcher = JinaReranker()
            print("Using Jina Reranker")
        else:
            self.semantic_searcher = InfinitySemanticSearcher()
            print("Using Infinity Reranker")

    async def process_sources(
        self,
        sources: SearchResult[Dict[str, Any]],
        num_elements: int,
        query: str,
        pro_mode: bool = False,
        min_sources: int = 0,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes search results by fetching, scraping, and reranking content.
        Yields intermediate status updates and the final processed sources dictionary.
        """
        processed_sources_data = sources.data if sources.data else {}
        try:
            print(
                f"[DEBUG] Starting process_sources with query: {query}, num_elements: {num_elements}, pro_mode: {pro_mode}, min_sources: {min_sources}"
            )

            if not sources.data:
                print("[ERROR] sources object does not have data")
                yield {"type": "error", "content": "No source data found"}
                return

            valid_sources = _get_valid_sources(sources, max(num_elements, min_sources))
            print(f"[DEBUG] Valid sources found: {len(valid_sources)}")

            if not valid_sources:
                print("[WARNING] No valid sources found to process")
                yield {"type": "status", "content": "No valid sources found to process."}

                yield {"type": "processed_sources", "data": processed_sources_data}
                return

            if not pro_mode and min_sources <= 1:
                wiki_sources = [(i, source) for i, source in valid_sources if "wikipedia.org" in source.get("link", "")]
                print(f"[DEBUG] Wikipedia sources found: {len(wiki_sources)}")

                if not wiki_sources:
                    if min_sources > 0:
                        print(f"[DEBUG] Using first {min_sources} sources as no Wikipedia sources found")
                        valid_sources = valid_sources[:min_sources]
                    else:
                        print("[DEBUG] No Wikipedia sources found and min_sources <= 1, will yield no sources")
                        yield {
                            "type": "status",
                            "content": "No Wikipedia source found and min_sources <= 1.",
                        }
                        yield {"type": "processed_sources", "data": processed_sources_data}
                        return
                else:
                    if min_sources <= 1:
                        print("[DEBUG] Using only the first Wikipedia source")
                        valid_sources = wiki_sources[:1]
                    else:
                        print(f"[DEBUG] Using Wikipedia source + {min_sources-1} other sources")
                        other_sources = [
                            (i, source) for i, source in valid_sources if "wikipedia.org" not in source.get("link", "")
                        ]
                        valid_sources = wiki_sources[:1] + other_sources[: min_sources - 1]
            elif pro_mode:

                print(f"[DEBUG] Pro mode: Using up to {num_elements} sources")
                valid_sources = valid_sources[:num_elements]
            elif min_sources > 0:

                print(f"[DEBUG] Using first {max(num_elements, min_sources)} sources (min_sources > 1)")
                valid_sources = valid_sources[: max(num_elements, min_sources)]

            links_to_scrape = [s[1].get("link") for s in valid_sources if s[1].get("link")]
            print(f"[DEBUG] Links to scrape: {links_to_scrape}")

            if not links_to_scrape:
                print("[ERROR] No valid links found in selected sources")
                yield {"type": "status", "content": "No valid links found in selected sources."}
                yield {"type": "processed_sources", "data": processed_sources_data}
                return

            yield {"type": "scraping_start", "urls": links_to_scrape}
            html_contents = await self._fetch_html_contents(links_to_scrape)
            print(f"[DEBUG] Fetched HTML contents for {len(html_contents)} URLs")
            yield {"type": "scraping_end"}

            async for update in self._update_sources_with_content(
                processed_sources_data, valid_sources, html_contents, query
            ):
                yield update

            print(
                f"[DEBUG] Completed processing sources, returning data with {len(processed_sources_data.get('organic', []))} organic results"
            )
            yield {"type": "processed_sources", "data": processed_sources_data}

        except Exception as e:
            print(f"[ERROR] Error in process_sources: {e}")
            yield {"type": "error", "content": f"Error processing sources: {e}"}

            yield {"type": "processed_sources", "data": processed_sources_data}

    async def _fetch_html_contents(self, links: List[str]) -> List[str]:
        # TODO: Check if scraper.scrape_many can yield progress
        raw_contents = await self.scraper.scrape_many(links)

        return [
            res.get("no_extraction").content if res and res.get("no_extraction") else ""
            for res in raw_contents.values()
        ]

    def _process_html_content(self, html: str, query: str) -> str:
        if not html:
            return ""
        try:
            documents = self.chunker.split_text(html)
            if not documents:
                return ""

            reranked_content = self.semantic_searcher.get_reranked_documents(query, documents, top_k=self.top_results)
            return reranked_content
        except Exception as e:
            print(f"Error in content processing: {e}")
            return ""

    async def _update_sources_with_content(
        self,
        sources_data: Dict[str, Any],
        valid_sources: List[Tuple[int, dict]],
        html_contents: List[str],
        query: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Updates the source dictionaries with processed HTML content.
        Yields status updates for each source being processed.
        """

        if len(html_contents) != len(valid_sources):
            print(
                f"Warning: Mismatch between valid sources ({len(valid_sources)}) and fetched HTML ({len(html_contents)})"
            )

            html_contents.extend([""] * (len(valid_sources) - len(html_contents)))

        for (original_index, source_dict), html in zip(valid_sources, html_contents):
            link = source_dict.get("link", "Unknown URL")
            yield {"type": "processing_source", "url": link}
            try:

                processed_content = await asyncio.to_thread(self._process_html_content, html, query)
            except Exception as e:
                print(f"Error running _process_html_content in thread for {link}: {e}")
                processed_content = ""

            if (
                "organic" in sources_data
                and isinstance(sources_data["organic"], list)
                and original_index < len(sources_data["organic"])
            ):

                target_dict = sources_data["organic"][original_index]
                if target_dict.get("link") == link:
                    target_dict["html"] = processed_content
                else:
                    print(f"Warning: Index mismatch updating source {link}")
                    source_dict["html"] = processed_content
            else:
                print(f"Warning: Could not find original source dict for index {original_index} and link {link}")
                source_dict["html"] = processed_content
