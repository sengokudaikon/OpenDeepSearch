from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from opendeepsearch.context_scraping.crawl4ai_scraper import WebScraper
from opendeepsearch.ranking_models.infinity_rerank import InfinitySemanticSearcher
from opendeepsearch.ranking_models.jina_reranker import JinaReranker
from opendeepsearch.ranking_models.chunker import Chunker
from opendeepsearch.serp_search.serp_search import SearchResult

@dataclass
class Source:
    link: str
    html: str = ""


def _get_valid_sources(sources: SearchResult[Dict[str, Any]], num_elements: int) -> List[Tuple[int, dict]]:
    # Make sure sources.data['organic'] exists and is a list
    if sources.data and 'organic' in sources.data:
        organic_results = sources.data['organic']
        if isinstance(organic_results, list):
            return [(i, source) for i, source in enumerate(organic_results[:num_elements]) if source]
    print("Warning: Could not extract valid sources from the search results")
    return []


class SourceProcessor:
    def __init__(
        self,
        top_results: int = 5,
        strategies: List[str] = ["no_extraction"],
        filter_content: bool = True,
        reranker: str = "jina"
    ):
        self.strategies = strategies
        self.filter_content = filter_content
        self.scraper = WebScraper(
            strategies=self.strategies,
            filter_content=self.filter_content
        )
        self.top_results = top_results
        self.chunker = Chunker()

        # Initialize the appropriate reranker
        if reranker.lower() == "jina":
            self.semantic_searcher = JinaReranker()
            print("Using Jina Reranker")
        else:  # default to infinity
            self.semantic_searcher = InfinitySemanticSearcher()
            print("Using Infinity Reranker")

    async def process_sources(
        self,
        sources: SearchResult[Dict[str, Any]],
        num_elements: int,
        query: str,
        pro_mode: bool = False,
        min_sources: int = 0
    ) -> Dict[str, Any]:
        try:
            if not sources.data:
                print("Warning: sources object does not have data")
                return {}

            valid_sources = _get_valid_sources(sources, max(num_elements, min_sources))
            if not valid_sources:
                return sources.data

            if not pro_mode and min_sources <= 1:
                # Check if there's a Wikipedia article among valid sources
                wiki_sources = [(i, source) for i, source in valid_sources
                              if 'wikipedia.org' in source['link']]
                if not wiki_sources:
                    # If min_sources is set and we don't have Wikipedia, use other sources
                    if min_sources > 0:
                        valid_sources = valid_sources[:min_sources]
                    else:
                        return sources.data
                else:
                    # If Wikipedia article exists and min_sources <= 1, only process that
                    # Otherwise, include Wikipedia and other sources to meet min_sources
                    if min_sources <= 1:
                        valid_sources = wiki_sources[:1]  # Take only the first Wikipedia source
                    else:
                        # Prioritize Wikipedia but include other sources to meet min_sources
                        other_sources = [(i, source) for i, source in valid_sources
                                      if 'wikipedia.org' not in source['link']]
                        valid_sources = wiki_sources[:1] + other_sources[:min_sources-1]
            elif min_sources > 0:
                # Ensure we have at least min_sources sources
                valid_sources = valid_sources[:max(num_elements, min_sources)]

            html_contents = await self._fetch_html_contents([s[1]['link'] for s in valid_sources])
            return self._update_sources_with_content(sources.data, valid_sources, html_contents, query)
        except Exception as e:
            print(f"Error in process_sources: {e}")
            return sources.data or {}

    async def _fetch_html_contents(self, links: List[str]) -> List[str]:
        raw_contents = await self.scraper.scrape_many(links)
        return [x['no_extraction'].content for x in raw_contents.values()]

    def _process_html_content(self, html: str, query: str) -> str:
        if not html:
            return ""
        try:
            # Split the HTML content into chunks
            documents = self.chunker.split_text(html)

            # Rerank the chunks based on the query
            reranked_content = self.semantic_searcher.get_reranked_documents(
                query,
                documents,
                top_k=self.top_results
            )

            return reranked_content

        except Exception as e:
            print(f"Error in content processing: {e}")
            return ""

    def _update_sources_with_content(
        self,
        sources: Dict[str, Any],
        valid_sources: List[Tuple[int, dict]],
        html_contents: List[str],
        query: str
    ) -> Dict[str, Any]:
        for (_, source), html in zip(valid_sources, html_contents):
            source['html'] = self._process_html_content(html, query)
        return sources