# pylint: disable=line-too-long
# flake8: noqa: E501

import json
from typing import List, Optional  # Added Optional

import requests
import torch

from opendeepersearch.config import config  # Added config import
from opendeepersearch.ranking_models.base_reranker import BaseSemanticSearcher


class InfinitySemanticSearcher(BaseSemanticSearcher):
    """
    A semantic reranking model that uses the Infinity Embedding API for text embeddings.

    This class provides methods to rerank documents based on their semantic similarity
    to queries using embeddings from the Infinity API. The API endpoint expects to receive
    text inputs and returns high-dimensional embeddings that capture semantic meaning.

    The default model used is 'Alibaba-NLP/gte-Qwen2-7B-instruct', but other models
    available through the Infinity API can be specified.

    Attributes:
        embedding_endpoint (str): URL of the Infinity Embedding API endpoint
        model_name (str): Name of the embedding model to use

    Example:
        ```python
        reranker = SemanticSearch(
            embedding_endpoint="http://localhost:7997/embeddings",
            model_name="Alibaba-NLP/gte-Qwen2-7B-instruct"
        )

        documents = [
            "Munich is in Germany.",
            "The sky is blue."
        ]

        results = reranker.rerank(
            query="What color is the sky?",
            documents=documents,
            top_k=1
        )
        ```
    """

    def __init__(
        self,
        embedding_endpoint: Optional[str] = None,  # Changed to Optional[str] = None
        model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        instruction_prefix: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    ):
        """
        Initialize the semantic search engine with Infinity Embedding API settings.

        Args:
            embedding_endpoint: URL of the Infinity Embedding API endpoint
            model_name: Name of the embedding model available in Infinity API
            instruction_prefix: Prefix to add to queries for better search relevance
        """
        # Use provided endpoint or fall back to config default
        self.embedding_endpoint = embedding_endpoint or str(config.infinity.endpoint)
        self.model_name = model_name
        self.instruction_prefix = instruction_prefix

    def _get_embeddings(self, texts: List[str], embedding_type: str = "query") -> torch.Tensor:
        """
        Get embeddings for a list of texts using the Infinity API.
        """
        max_texts = 2048
        if len(texts) > max_texts:
            import warnings

            warnings.warn(f"Number of texts ({len(texts)}) exceeds maximum of {max_texts}. List will be truncated.")
            texts = texts[:max_texts]

        formatted_texts = [self.instruction_prefix + text if embedding_type == "query" else text for text in texts]

        response = requests.post(
            self.embedding_endpoint, json={"model": self.model_name, "input": formatted_texts}, timeout=10
        )

        content_str = response.content.decode("utf-8")
        content_json = json.loads(content_str)
        return torch.tensor([item["embedding"] for item in content_json["data"]])
