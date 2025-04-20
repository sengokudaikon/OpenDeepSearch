import argparse
import asyncio
import importlib.util
import os
import socket
import sys
from contextlib import closing
from typing import Any, AsyncGenerator, Dict, List, Tuple

import gradio as gr
from dotenv import load_dotenv

if importlib.util.find_spec("gradio") is None:
    print("Error: The 'gradio' package is required for this script.")
    print("Install it with: pip install -e .[gradio-demo] or pip install -e .[demo] or pip install gradio")
    sys.exit(1)

from opendeepersearch import OpenDeepSearchTool

load_dotenv()


parser = argparse.ArgumentParser(description="Run the Gradio demo with custom models")
parser.add_argument(
    "--model-name",
    default=os.getenv(
        "LITELLM_SEARCH_MODEL_ID",
        os.getenv("LITELLM_MODEL_ID", "openrouter/google/gemini-2.0-flash-001"),
    ),
    help="Model name for search",
)


parser.add_argument(
    "--reranker",
    choices=["jina", "infinity"],
    default="jina",
    help="Reranker to use (jina or infinity)",
)
parser.add_argument(
    "--search-provider",
    choices=["serper", "searxng"],
    default="serper",
    help="Search provider to use (serper or searxng)",
)
parser.add_argument(
    "--searxng-instance",
    default=os.getenv("SEARXNG_INSTANCE_URL"),
    help="SearXNG instance URL (required if search-provider is searxng)",
)
parser.add_argument("--searxng-api-key", default=os.getenv("SEARXNG_API_KEY"), help="SearXNG API key (optional)")
parser.add_argument(
    "--serper-api-key",
    default=os.getenv("SERPER_API_KEY"),
    help="Serper API key (optional, will use SERPER_API_KEY env var if not provided)",
)
parser.add_argument(
    "--openai-base-url",
    default=os.getenv("OPENAI_BASE_URL"),
    help="OpenAI API base URL (optional, will use OPENAI_BASE_URL env var if not provided)",
)
parser.add_argument("--server-port", type=int, default=7860, help="Port to run the Gradio server on")

parser.add_argument(
    "--pro-mode",
    action="store_true",
    help="Enable deep research mode with iterative search workflow and transparent reasoning",
)
parser.add_argument("--min-sources", type=int, default=1, help="Minimum number of sources to include")
parser.add_argument(
    "--max-sources",
    type=int,
    default=3,
    help="Maximum number of sources to include (normal mode default: 3)",
)
parser.add_argument("--max-iterations", type=int, default=3, help="Maximum number of search iterations in pro mode")
parser.add_argument(
    "--max-sources-per-iteration",
    type=int,
    default=10,
    help="Maximum sources to process per iteration in pro mode (default: 10)",
)

args = parser.parse_args()


if args.search_provider == "searxng" and not args.searxng_instance:
    parser.error("--searxng-instance is required when using --search-provider=searxng")


if args.openai_base_url:
    os.environ["OPENAI_BASE_URL"] = args.openai_base_url


search_tool = OpenDeepSearchTool(
    model_name=args.model_name,
    reranker=args.reranker,
    search_provider=args.search_provider,
    serper_api_key=args.serper_api_key,
    searxng_instance_url=args.searxng_instance,
    searxng_api_key=args.searxng_api_key,
)

search_tool.setup()


async def format_stream_output(stream_chunk: Dict[str, Any]) -> str:
    """Formats a chunk from the stream generator into Markdown for display."""
    chunk_type = stream_chunk.get("type")
    content = stream_chunk.get("content")
    sources = stream_chunk.get("sources")
    answer = stream_chunk.get("answer")
    url = stream_chunk.get("url")
    urls = stream_chunk.get("urls")
    error = stream_chunk.get("error")
    warning = stream_chunk.get("warning")

    console_output = f"[DEBUG] Chunk type: {chunk_type}"
    if content:
        console_output += f" | Content: {content[:50]}..." if len(str(content)) > 50 else f" | Content: {content}"
    elif answer:
        console_output += f" | Answer: {answer[:50]}..." if len(str(answer)) > 50 else f" | Answer: {answer}"
        if sources:
            console_output += f" | Sources: {len(sources)}"
    elif url:
        console_output += f" | URL: {url}"
    elif urls:
        console_output += f" | URLs: {len(urls)} items"
    print(console_output)

    output = ""
    if chunk_type == "status":
        output = f"⏳ *{content}*\n"
    elif chunk_type == "thought":

        formatted_content = content.replace("\n", "<br>")
        output = f"""<details>
<summary><b>🤔 Thought Process</b></summary>
<div style="padding: 10px; border-left: 3px solid #ddd; margin: 10px 0;">
{formatted_content}
</div>
</details>"""
    elif chunk_type == "sources_found":
        output = f"📚 Found {len(content)} potential sources:\n"
        output += "\n".join([f"{i+1}. {src}" for i, src in enumerate(content)])
    elif chunk_type == "scraping_start":
        output = f"🕸️ Starting to scrape {len(urls)} URLs..."
    elif chunk_type == "scraping_end":
        output = "✅ Finished scraping."
    elif chunk_type == "processing_source":
        output = f"📄 Processing source: {url}"
    elif chunk_type == "final_answer":

        try:
            if answer:
                print(f"[DEBUG] Received final answer: {answer[:100]}...")
                output = f"""## Final Answer

{answer}

"""
                if sources:
                    print(f"[DEBUG] Adding {len(sources)} sources to final answer")
                    output += "**Sources Used:**\n"
                    for i, s in enumerate(sources):
                        try:
                            title = s.get("title") or s.get("link", "Unknown source")
                            link = s.get("link", "#")
                            output += f"{i+1}. [{title}]({link})\n"
                        except Exception as e:
                            print(f"[DEBUG] Error formatting source {i}: {e}")
                            output += f"{i+1}. [Source {i+1}](#)\n"
            else:

                print(
                    "[DEBUG] No 'answer' field found in final_answer chunk, trying to extract from content or other fields"
                )
                if content:
                    output = f"""## Final Answer

{content}

"""
                elif isinstance(stream_chunk, dict):

                    for key, value in stream_chunk.items():
                        if isinstance(value, str) and len(value) > 50:
                            print(f"[DEBUG] Using value from key '{key}' as answer")
                            output = f"""## Final Answer

{value}

"""
                            break

                if not output:
                    output = """## Final Answer

The search has completed, but no formatted answer could be extracted from the results.
Please check the sources above for relevant information.

"""
        except Exception as e:
            print(f"[DEBUG] Error formatting final answer: {e}")
            import traceback

            traceback.print_exc()
            output = """## Final Answer

An error occurred while formatting the final answer. Please check the sources above for relevant information.

"""
    elif chunk_type == "error":
        output = f"❌ **Error:** {error}"
    elif chunk_type == "warning":
        output = f"⚠️ **Warning:** {warning}"
    else:
        output = f"Unknown stream type: {chunk_type}"

    return output + "\n\n"


async def run_search(query: str, history: List[Tuple[str, str]]) -> AsyncGenerator[List[Tuple[str, str]], None]:
    """Runs the search tool's stream_forward and yields updates to the Gradio chatbot."""
    history.append((query, ""))
    current_response = ""
    final_answer_received = False
    final_answer_content = ""

    effective_max_sources = args.max_sources
    effective_max_sources_per_iteration = args.max_sources_per_iteration if args.pro_mode else 3

    try:

        all_chunks = []

        async for chunk in search_tool.stream_forward(
            query=query,
            min_sources=args.min_sources,
            max_sources=effective_max_sources,
            pro_mode=args.pro_mode,
            max_iterations=args.max_iterations,
            max_sources_per_iteration=effective_max_sources_per_iteration,
        ):
            all_chunks.append(chunk)
            chunk_type = chunk.get("type")

            if chunk_type != "final_answer":
                formatted_chunk = await format_stream_output(chunk)
                current_response += formatted_chunk
                history[-1] = (query, current_response)
                yield history
                await asyncio.sleep(0.01)
            else:

                final_answer_received = True
                final_answer_content = await format_stream_output(chunk)

        if final_answer_received and final_answer_content:
            print("[DEBUG] Appending final answer to response")
            current_response += final_answer_content
            history[-1] = (query, current_response)
            yield history

        elif not final_answer_received:
            print("[DEBUG] No final answer received, adding completion message")
            completion_message = "\n\n## Final Answer\n\nThe search has completed, but no final answer was generated. Please review the sources above for relevant information.\n\n"
            current_response += completion_message
            history[-1] = (query, current_response)
            yield history

    except Exception as e:
        error_message = f"\n\n❌ **An error occurred during search:** {str(e)}"
        print(f"[DEBUG] Error in run_search: {str(e)}")
        import traceback

        traceback.print_exc()
        current_response += error_message
        history[-1] = (query, current_response)
        yield history


def find_free_port(start_port):
    """Find a free port starting from start_port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        for port in range(start_port, start_port + 100):
            try:
                s.bind(("127.0.0.1", port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                continue
    raise OSError(f"Could not find a free port in range {start_port}-{start_port+100}")


free_port = find_free_port(args.server_port)
if free_port != args.server_port:
    print(f"Port {args.server_port} is in use. Using port {free_port} instead.")


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    mode_status = "Deep Research Mode" if args.pro_mode else "Standard Search Mode"

    effective_max_sources = args.max_sources
    effective_max_sources_per_iteration = args.max_sources_per_iteration if args.pro_mode else 3

    gr.Markdown(
        f"""
    # OpenDeepSearch Agent Demo
    Using search model: `{args.model_name}` | Reranker: `{args.reranker}` | Search Provider: `{args.search_provider}`

    **Configuration:** {mode_status} | Min Sources: {args.min_sources} | Max Sources: {effective_max_sources}
    {f"| Max Iterations: {args.max_iterations} | Max Sources Per Iteration: {effective_max_sources_per_iteration}" if args.pro_mode else ""}
    """
    )
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=600,
        avatar_images=(
            None,
            (os.path.join(os.path.dirname(__file__), "assets/sentient-logo-narrow.png")),
        ),
        render_markdown=True,
        show_copy_button=True,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter your query and press enter",
            container=False,
        )

    txt.submit(run_search, [txt, chatbot], chatbot)
    txt.submit(lambda: "", None, txt)


def main():
    print(f"Launching Gradio server on http://127.0.0.1:{free_port}")
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=free_port, share=True)

if __name__ == "__main__":
    main()
