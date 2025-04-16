# 🔍 OpenDeeperSearch: Advanced Search with Open-source Reasoning Models 🚀

> **Note:** This project is a significantly modified fork of [sentient-agi/OpenDeepSearch](https://github.com/sentient-agi/OpenDeepSearch) with substantial enhancements and new features.

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
    <h3>Based on research from the paper:</h3>
    <h4 align="center">
        <a href="https://arxiv.org/pdf/2503.20201">Open Deep Search: Democratizing Search with Open-source Reasoning Agents</a>
    </h4>
</div>

<hr>

## Description 📝

OpenDeeperSearch is a lightweight yet powerful search tool designed for seamless integration with AI agents. It enables deep web search and retrieval, optimized for use with Hugging Face's **[SmolAgents](https://github.com/huggingface/smolagents)** ecosystem. This version includes significant improvements and new features developed by Daniil Zavrin.

<div align="center">
    <img src="./assets/evals.png" alt="Evaluation Results" width="80%"/>
</div>

- **Performance**: ODS performs on par with closed source search alternatives on single-hop queries such as [SimpleQA](https://openai.com/index/introducing-simpleqa/) 🔍.
- **Advanced Capabilities**: ODS performs much better than closed source search alternatives on multi-hop queries such as [FRAMES bench](https://huggingface.co/datasets/google/frames-benchmark) 🚀.

## Table of Contents 📑

- [🔍 OpenDeeperSearch: Advanced Search with Open-source Reasoning Models 🚀](#-enhanced-opendeepsearch-advanced-search-with-open-source-reasoning-models-)
  - [Description 📝](#description-)
  - [Table of Contents 📑](#table-of-contents-)
  - [Features ✨](#features-)
  - [Installation 📚](#installation-)
  - [Setup](#setup)
  - [Usage ️](#usage-️)
    - [Using OpenDeepSearch Standalone 🔍](#using-opendeepsearch-standalone-)
    - [Running the Gradio Demo 🖥️](#running-the-gradio-demo-️)
    - [Integrating with SmolAgents \& LiteLLM 🤖⚙️](#integrating-with-smolagents--litellm-️)
    - [ReAct agent with math and search tools 🤖⚙️](#react-agent-with-math-and-search-tools-️)
  - [Search Modes 🔄](#search-modes-)
    - [Default Mode ⚡](#default-mode-)
    - [Pro Mode 🔍](#pro-mode-)
  - [Development](#development)
  - [Acknowledgments 💡](#acknowledgments-)
  - [Citation](#citation)
  - [License](#license)
  - [Contact 📩](#contact-)
  - [MCP Server 🔌](#mcp-server-)

## Features ✨

- **Semantic Search** 🧠: Leverages **[Crawl4AI](https://github.com/unclecode/crawl4ai)** and semantic search rerankers (such as [Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/tree/main) and [Jina AI](https://jina.ai/)) to provide in-depth results
- **Two Modes of Operation** ⚡:
  - **Default Mode**: Quick and efficient search with minimal latency.
  - **Pro Mode (Deep Search)**: More in-depth and accurate results at the cost of additional processing time.
- **Optimized for AI Agents** 🤖: Works seamlessly with **SmolAgents** like `CodeAgent`.
- **Fast and Lightweight** ⚡: Designed for speed and efficiency with minimal setup.
- **Extensible** 🔌: Easily configurable to work with different models and APIs.
- **Streaming Support** 🌊: Added support for streaming responses in Gradio interface.
- **Enhanced Error Handling** 🛡️: Improved error handling and recovery mechanisms.
- **Modern Development Workflow** 🛠️: Comprehensive tooling for development, testing, and deployment.

## Installation 📚

To install OpenDeeperSearch, run:

```bash
pip install -e . #you can also use: uv pip install -e .
pip install -r requirements.txt #you can also use: uv pip install -r requirements.txt
```

Note: you must have `torch` installed.
Note: using `uv` instead of regular `pip` makes life much easier!

### Using PDM (Alternative Package Manager) 📦

You can also use PDM as an alternative package manager for OpenDeepSearch. PDM is a modern Python package and dependency manager supporting the latest PEP standards.

```bash
# Install PDM if you haven't already
curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -

# Initialize a new PDM project
pdm init

# Install OpenDeepSearch and its dependencies
pdm install

# Activate the virtual environment
eval "$(pdm venv activate)"
```

PDM offers several advantages:
- Lockfile support for reproducible installations
- PEP 582 support (no virtual environment needed)
- Fast dependency resolution
- Built-in virtual environment management

## Setup

1. **Choose a Search Provider**:
   - **Option 1: Serper.dev**: Get **free 2500 credits** and add your API key.
     - Visit [serper.dev](https://serper.dev) to create an account.
     - Retrieve your API key and store it as an environment variable:

     ```bash
     export SERPER_API_KEY='your-api-key-here'
     ```

   - **Option 2: SearXNG**: Use a self-hosted or public SearXNG instance.
     - Specify the SearXNG instance URL when initializing OpenDeepSearch.
     - Optionally provide an API key if your instance requires authentication:

     ```bash
     export SEARXNG_INSTANCE_URL='https://your-searxng-instance.com'
     export SEARXNG_API_KEY='your-api-key-here'  # Optional
     ```

2. **Choose a Reranking Solution**:
   - **Quick Start with Jina**: Sign up at [Jina AI](https://jina.ai/) to get an API key for immediate use
   - **Self-hosted Option**: Set up [Infinity Embeddings](https://github.com/michaelfeil/infinity) server locally with open source models such as [Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/tree/main)
   - For more details on reranking options, see our [Rerankers Guide](src/opendeepersearch/ranking_models/README.md)

3. **Set up LiteLLM Provider**:
   - Choose a provider from the [supported list](https://docs.litellm.ai/docs/providers/), including:
     - OpenAI
     - Anthropic
     - Google (Gemini)
     - Microsoft Azure OpenAI
     - OpenRouter
     - Fireworks AI
     - DeepSeek
     - HuggingFace
     - And many more! See the full [LiteLLM supported providers list](https://docs.litellm.ai/docs/providers/).
   - Set your chosen provider's API key as an environment variable:
   ```bash
   export <PROVIDER>_API_KEY='your-api-key-here'  # e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, AZURE_API_KEY, DEEPSEEK_API_KEY
   ```
   - For specific providers like Azure, additional variables are needed (LiteLLM automatically detects these):
   ```bash
   export AZURE_API_BASE='https://your-resource-name.openai.azure.com/'
   export AZURE_API_VERSION='2024-02-01' # Or your specific API version
   # AZURE_DEPLOYMENT_ID is often passed via the model name, e.g., "azure/your-deployment-id"
   ```
   - For OpenAI-compatible endpoints (like self-hosted models), you can set:
   ```bash
   export OPENAI_API_KEY='your_api_key_if_needed'
   export OPENAI_BASE_URL='https://your-custom-endpoint.com/v1' # Note the /v1 suffix usually required
   ```
   - You can set default LiteLLM model IDs for different tasks:
   ```bash
   # General default model (fallback for all tasks)
   export LITELLM_MODEL_ID='openrouter/google/gemini-2.0-flash-001'

   # Task-specific models
   export LITELLM_SEARCH_MODEL_ID='openrouter/google/gemini-2.0-flash-001'  # For search tasks
   export LITELLM_ORCHESTRATOR_MODEL_ID='openrouter/google/gemini-2.0-flash-001'  # For agent orchestration
   export LITELLM_EVAL_MODEL_ID='gpt-4o-mini'  # For evaluation tasks
   ```
   - When initializing OpenDeepSearch, you can specify your chosen model using the provider's format (this will override the environment variables):
   ```python
   search_agent = OpenDeepSearchTool(model="provider/model-name") # e.g., "gemini/gemini-pro", "azure/your-deployment-id", "deepseek/deepseek-chat"
   ```

## Usage ️

You can use OpenDeepSearch independently or integrate it with **SmolAgents** for enhanced reasoning and code generation capabilities.

### Using OpenDeepSearch Standalone 🔍

```python
from opendeepersearch import OpenDeepSearchTool
import os

# Set environment variables for API keys
os.environ["SERPER_API_KEY"] = "your-serper-api-key-here"  # If using Serper
# Or for SearXNG
# os.environ["SEARXNG_INSTANCE_URL"] = "https://your-searxng-instance.com"
# os.environ["SEARXNG_API_KEY"] = "your-api-key-here"  # Optional

os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key-here"
os.environ["JINA_API_KEY"] = "your-jina-api-key-here"

# Using Serper (default)
search_agent = OpenDeepSearchTool(
    model_name="openrouter/google/gemini-2.0-flash-001",
    reranker="jina"
)

# Or using SearXNG
# search_agent = OpenDeepSearchTool(
#     model_name="openrouter/google/gemini-2.0-flash-001",
#     reranker="jina",
#     search_provider="searxng",
#     searxng_instance_url="https://your-searxng-instance.com",
#     searxng_api_key="your-api-key-here"  # Optional
# )

if not search_agent.is_initialized:
    search_agent.setup()

query = "Fastest land animal?"

# Basic search
result = search_agent.forward(query)
print(result)

# Search with minimum number of sources
result_with_sources = search_agent.forward(query, min_sources=3)
print(f"Result with at least 3 sources: {result_with_sources}")
```

### Running the Demos 🖥️

OpenDeeperSearchoffers two different demo interfaces:

1. **Gradio Demo**: A chat-like interface with streaming responses
2. **Web Demo**: A simple web interface using FastAPI and HTMX

#### Using Just Commands

The easiest way to run the demos is using the provided Just commands:

```bash
# Run the Gradio demo
just demo

# Run the Web demo
just web-demo

# Install all demo dependencies (without running any demo)
just all-demos
```

#### Manual Installation and Running

Alternatively, you can install the dependencies manually:

```bash
# For the Gradio demo
uv pip install -e .[gradio-demo]
python gradio_demo.py

# For the Web demo
uv pip install -e .[web-demo]
python simple_web_demo.py

# For both demos
uv pip install -e .[demo]
```

Both demos will automatically check if the required dependencies are installed and provide instructions if anything is missing.

#### Customizing the Demos

You can customize either demo with similar command-line arguments:

```bash
# Example for Gradio demo
python gradio_demo.py --model-name "openrouter/google/gemini-2.0-flash-001" --reranker "jina"

# Example for Web demo
python simple_web_demo.py --model-name "openrouter/google/gemini-2.0-flash-001" --reranker "jina"

# Using SearXNG and Infinity reranker (works with either demo)
python gradio_demo.py --model-name "openrouter/google/gemini-2.0-flash-001" --reranker "infinity" \
  --search-provider "searxng" --searxng-instance "https://your-searxng-instance.com" \
  --searxng-api-key "your-api-key-here"  # Optional
```

Available options:
- `--model-name`: LLM model to use for search (defaults to `LITELLM_SEARCH_MODEL_ID` or `LITELLM_MODEL_ID` env var, or `openrouter/google/gemini-2.0-flash-001`).
- `--orchestrator-model`: LLM model for the agent orchestrator (defaults to `LITELLM_ORCHESTRATOR_MODEL_ID` or `LITELLM_MODEL_ID` env var, or `openrouter/google/gemini-2.0-flash-001`).
- `--reranker`: Reranker to use (`jina` or `infinity`, default: `jina`).
- `--search-provider`: Search provider to use (`serper` or `searxng`, default: `serper`).
- `--searxng-instance`: SearXNG instance URL (required if using `searxng`, defaults to `SEARXNG_INSTANCE_URL` env var).
- `--searxng-api-key`: SearXNG API key (optional, defaults to `SEARXNG_API_KEY` env var).
- `--serper-api-key`: Serper API key (optional, defaults to `SERPER_API_KEY` env var).
- `--openai-base-url`: OpenAI API base URL (optional, defaults to `OPENAI_BASE_URL` env var).
- `--server-port`: Port to run the Gradio server on (default: 7860).

### Integrating with SmolAgents & LiteLLM 🤖⚙️

```python
from opendeepersearch import OpenDeepSearchTool
from smolagents import CodeAgent, LiteLLMModel
import os

# Set environment variables for API keys
os.environ["SERPER_API_KEY"] = "your-serper-api-key-here"  # If using Serper
# Or for SearXNG
# os.environ["SEARXNG_INSTANCE_URL"] = "https://your-searxng-instance.com"
# os.environ["SEARXNG_API_KEY"] = "your-api-key-here"  # Optional

os.environ["OPENROUTER_API_KEY"] = "your-openrouter-api-key-here"
os.environ["JINA_API_KEY"] = "your-jina-api-key-here"

# Using Serper (default)
search_agent = OpenDeepSearchTool(
    model_name="openrouter/google/gemini-2.0-flash-001",
    reranker="jina"
)

# Or using SearXNG
# search_agent = OpenDeepSearchTool(
#     model_name="openrouter/google/gemini-2.0-flash-001",
#     reranker="jina",
#     search_provider="searxng",
#     searxng_instance_url="https://your-searxng-instance.com",
#     searxng_api_key="your-api-key-here"  # Optional
# )

model = LiteLLMModel(
    "openrouter/google/gemini-2.0-flash-001",
    temperature=0.2
)

code_agent = CodeAgent(tools=[search_agent], model=model)
query = "How long would a cheetah at full speed take to run the length of Pont Alexandre III?"
result = code_agent.run(query)

print(result)
```
### ReAct agent with math and search tools 🤖⚙️

```python
from opendeepersearch import OpenDeepSearchTool
from opendeepersearch.wolfram_tool import WolframAlphaTool
from opendeepersearch.prompts import REACT_PROMPT
from smolagents import LiteLLMModel, ToolCallingAgent, Tool
import os

# Set environment variables for API keys
os.environ["SERPER_API_KEY"] = "your-serper-api-key-here"
os.environ["JINA_API_KEY"] = "your-jina-api-key-here"
os.environ["WOLFRAM_ALPHA_APP_ID"] = "your-wolfram-alpha-app-id-here"
os.environ["FIREWORKS_API_KEY"] = "your-fireworks-api-key-here"

model = LiteLLMModel(
    "fireworks_ai/llama-v3p1-70b-instruct",  # Your Fireworks Deepseek model
    temperature=0.7
)
search_agent = OpenDeepSearchTool(model_name="fireworks_ai/llama-v3p1-70b-instruct", reranker="jina") # Set reranker to "jina" or "infinity"

# Initialize the Wolfram Alpha tool
wolfram_tool = WolframAlphaTool(app_id=os.environ["WOLFRAM_ALPHA_APP_ID"])

# Initialize the React Agent with search and wolfram tools
react_agent = ToolCallingAgent(
    tools=[search_agent, wolfram_tool],
    model=model,
    prompt_templates=REACT_PROMPT # Using REACT_PROMPT as system prompt
)

# Example query for the React Agent
query = "What is the distance, in metres, between the Colosseum in Rome and the Rialto bridge in Venice"
result = react_agent.run(query)

print(result)
```

## Search Modes 🔄

OpenDeepSearch offers two distinct search modes to balance between speed and depth, plus a parameter to control source diversity:

### Default Mode ⚡
- Uses SERP-based interaction for quick results
- Minimal processing overhead
- Ideal for single-hop, straightforward queries
- Fast response times
- Perfect for basic information retrieval

### Pro Mode 🔍
- Involves comprehensive web scraping
- Implements semantic reranking of results
- Includes advanced post-processing of data
- Slightly longer processing time
- Excels at:
  - Multi-hop queries
  - Complex search requirements
  - Detailed information gathering
  - Questions requiring cross-reference verification

### Source Control Parameters

#### min_sources Parameter
- Forces the model to search for a minimum number of unique sources
- Useful for ensuring information diversity and cross-verification
- Example usage: `search_agent.forward(query, min_sources=3)`
- Can be combined with either Default or Pro mode
- Helps prevent over-reliance on a single source (like Wikipedia)

#### max_sources Parameter
- Controls the maximum number of sources to process (default: 2)
- Balances between comprehensive information and processing time
- Example usage: `search_agent.forward(query, max_sources=5)`

## Development

This project uses a modern Python development workflow with comprehensive tooling. To set up the development environment:

```bash
# Create a fresh development environment
just fresh-start

# Run code quality checks
just quality-check

# Run tests
just test-cov
```

For more development commands, run:

```bash
just help
```

## Acknowledgments 💡

OpenDeeperSearch is built on the shoulders of great open-source projects:

- **[SmolAgents](https://huggingface.co/docs/smolagents/index)** 🤗 – Powers the agent framework and reasoning capabilities.
- **[Crawl4AI](https://github.com/unclecode/crawl4ai)** 🕷️ – Provides data crawling support.
- **[Infinity Embedding API](https://github.com/michaelfeil/infinity)** 🌍 – Powers semantic search capabilities.
- **[LiteLLM](https://www.litellm.ai/)** 🔥 – Used for efficient AI model integration.
- **Various Open-Source Libraries** 📚 – Enhancing search and retrieval functionalities.

## Citation

If you use `OpenDeeperSearch` in your works, please cite the original research paper:

```
@misc{alzubi2025opendeepsearchdemocratizing,
      title={Open Deep Search: Democratizing Search with Open-source Reasoning Agents},
      author={Salaheddin Alzubi and Creston Brooks and Purva Chiniya and Edoardo Contente and Chiara von Gerlach and Lucas Irwin and Yihan Jiang and Arda Kaz and Windsor Nguyen and Sewoong Oh and Himanshu Tyagi and Pramod Viswanath},
      year={2025},
      eprint={2503.20201},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.20201},
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact 📩

For questions or collaborations, open an issue or reach out to the maintainers.

## MCP Server 🔌

This project works with a Model Context Protocol (MCP) server located in the separate repository: [https://github.com/sengokudaikon/opendeepsearch_mcp](https://github.com/sengokudaikon/opendeepsearch_mcp). This server allows compatible clients (like Smithery, Claude Desktop, etc.) to interact with OpenDeepSearch programmatically.

For detailed setup and configuration instructions for the MCP server, please refer to its dedicated repository.
