import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, HttpUrl, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file if it exists
load_dotenv()


class SerperSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SERPER_")

    api_key: Optional[str] = None
    api_url: HttpUrl = Field(default="https://google.serper.dev/search")  # type: ignore[assignment]
    location: str = Field(default="us")
    timeout: int = Field(default=10, gt=0)


class SearXNGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEARXNG_")

    instance_url: Optional[HttpUrl] = None  # type: ignore[assignment]
    api_key: Optional[str] = None
    location: str = Field(default="all")
    timeout: int = Field(default=10, gt=0)


class JinaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JINA_")

    api_key: Optional[str] = None


class InfinitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFINITY_")

    endpoint: Optional[HttpUrl] = Field(default="http://localhost:7997/embeddings")


class WolframAlphaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WOLFRAM_ALPHA_")

    app_id: Optional[str] = None


class OpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENAI_")

    api_key: Optional[str] = None
    base_url: Optional[HttpUrl] = None


class AnthropicSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_")

    api_key: Optional[str] = None


class OpenRouterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OPENROUTER_")

    api_key: Optional[str] = None


class FireworksSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIREWORKS_")

    api_key: Optional[str] = None


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEMINI_")

    api_key: Optional[str] = None


class AzureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_")

    api_base: Optional[HttpUrl] = None
    api_key: Optional[str] = None
    api_version: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def check_azure_config_complete(self):
        api_base = self.api_base
        api_key = self.api_key
        api_version = self.api_version

        provided_values = [v for v in [api_base, api_key, api_version] if v is not None]

        if 0 < len(provided_values) < 3:
            raise ValueError(
                "Azure configuration requires all of AZURE_API_BASE, AZURE_API_KEY, "
                "and AZURE_API_VERSION to be set if any one is provided."
            )
        return self


class LiteLLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LITELLM_")

    model_id: str = Field(default="openrouter/google/gemini-2.0-flash-001")
    search_model_id: str = Field(default="openrouter/google/gemini-2.0-flash-001")
    orchestrator_model_id: str = Field(default="openrouter/google/gemini-2.0-flash-001")
    eval_model_id: str = Field(default="gpt-4o-mini")


class LLMGenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.3, ge=0.0, le=1.0)


class SearchSettings(BaseSettings):
    min_sources: int = Field(default=2, ge=1)
    max_sources: int = Field(default=10, ge=1)
    pro_mode: bool = Field(default=False)
    max_iterations: int = Field(default=3, ge=1)
    max_sources_per_iteration: int = Field(default=10, ge=1)

    num_serp_results_fetch: int = Field(default=30, ge=1)
    num_sources_pre_filter: int = Field(default=10, ge=1)
    pre_filtering_model_id: Optional[str] = Field(default=None)

    @field_validator("max_sources")
    @classmethod
    def max_sources_ge_min_sources(cls, v: int, info) -> int:
        """
        Ensure max_sources is not less than min_sources.
        """
        min_sources = info.data.get("min_sources")
        if isinstance(min_sources, int) and v < min_sources:
            raise ValueError("max_sources must be greater than or equal to min_sources")
        return v


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNK_")

    size: int = Field(default=150, gt=0)
    overlap: int = Field(default=50, ge=0)

    @field_validator("overlap")
    @classmethod
    def overlap_lt_size(cls, v: int, info) -> int:
        """
        Ensure overlap is less than size.
        """
        size = info.data.get("size")
        if isinstance(size, int) and v >= size:
            raise ValueError("overlap must be less than size")
        return v


class SourceProcessingSettings(BaseSettings):
    top_results: int = Field(default=5, ge=1)
    filter_content: bool = Field(default=True)


class AppConfig(BaseSettings):
    """Main application configuration."""

    # Search Providers
    serper: SerperSettings = Field(default_factory=SerperSettings)
    searxng: SearXNGSettings = Field(default_factory=SearXNGSettings)

    # Rerankers
    jina: JinaSettings = Field(default_factory=JinaSettings)
    infinity: InfinitySettings = Field(default_factory=InfinitySettings)

    # Additional Tools
    wolfram_alpha: WolframAlphaSettings = Field(default_factory=WolframAlphaSettings)

    # LLM Providers
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    fireworks: FireworksSettings = Field(default_factory=FireworksSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    azure: AzureSettings = Field(default_factory=AzureSettings)

    # Model Configuration
    litellm: LiteLLMSettings = Field(default_factory=LiteLLMSettings)

    # LLM Generation Parameters
    llm_generation: LLMGenerationSettings = Field(default_factory=LLMGenerationSettings)

    # Search Configuration
    search: SearchSettings = Field(default_factory=SearchSettings)

    # Text Chunking Configuration
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)

    # Source Processing Configuration
    source_processing: SourceProcessingSettings = Field(default_factory=SourceProcessingSettings)

    # Add a validator to ensure at least one search provider API key or URL is set if used
    @field_validator("serper", "searxng")
    @classmethod
    def check_search_provider_config(cls, v, info):
        if info.field_name == "serper" and v.api_key is None and os.getenv("SERPER_API_KEY"):
            pass  # Allow None if explicitly set or not provided
        elif info.field_name == "searxng" and v.instance_url is None and os.getenv("SEARXNG_INSTANCE_URL"):
            pass  # Allow None if explicitly set or not provided
        # No explicit error needed here, components using these should check for required fields
        return v


# Global config instance
try:
    config = AppConfig()
except ValidationError as e:
    print(f"Configuration Error:\n{e}")
    raise e
