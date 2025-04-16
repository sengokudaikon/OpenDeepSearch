import argparse
import asyncio
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from opendeepersearch import OpenDeepSearchTool

load_dotenv()


parser = argparse.ArgumentParser(description="Run the OpenDeepSearch Web Demo")
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
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
args = parser.parse_args()


app = FastAPI(title="OpenDeepSearch Web Demo")


search_tool = OpenDeepSearchTool(
    model_name=args.model_name,
    reranker=args.reranker,
    search_provider=args.search_provider,
    serper_api_key=os.getenv("SERPER_API_KEY"),
)

search_tool.setup()


templates = Jinja2Templates(directory="templates")


os.makedirs("templates", exist_ok=True)


with open("templates/index.html", "w") as f:
    f.write(
        """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenDeepSearch Web Demo</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
        }
        .search-container {
            margin-bottom: 20px;
        }
        input {
            width: 80%;
            padding: 10px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            padding: 10px 15px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #2980b9;
        }
        .results {
            margin-top: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            min-height: 200px;
        }
        .stream-output {
            white-space: pre-wrap;
        }
        .thought-process {
            border-left: 3px solid #3498db;
            padding-left: 10px;
            margin: 10px 0;
            color: #555;
        }
        .final-answer {
            font-weight: bold;
            margin-top: 20px;
        }
        .source-list {
            margin-top: 10px;
        }
        .source-item {
            margin-bottom: 5px;
        }
        .loading-indicator {
            display: none;
            color: #3498db;
            margin: 10px 0;
        }
        .htmx-request .loading-indicator {
            display: block;
        }
    </style>
</head>
<body>
    <h1>OpenDeepSearch Web Demo</h1>
    <p>Ask a question and see the streaming search results</p>

    <div class="search-container">
        <form hx-post="/search"
              hx-target="#search-results"
              hx-trigger="submit"
              hx-swap="innerHTML"
              hx-indicator=".loading-indicator">
            <input type="text" name="query" placeholder="Enter your question..." required>
            <button type="submit">Search</button>
        </form>
        <div class="loading-indicator">Searching... Please wait</div>
    </div>

    <div id="search-results" class="results">
        <p>Results will appear here...</p>
    </div>

</body>
</html>
    """
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def format_stream_chunk(chunk):
    chunk_type = chunk.get("type")
    content = chunk.get("content", "")
    sources = chunk.get("sources", [])
    answer = chunk.get("answer", "")
    url = chunk.get("url", "")
    urls = chunk.get("urls", [])

    if chunk_type == "status":
        return f"<p><i>⏳ {content}</i></p>"
    elif chunk_type == "thought":
        return f"""
            <div class="thought-process">
                <details>
                    <summary>🤔 Thought Process</summary>
                    <div>{content}</div>
                </details>
            </div>
        """
    elif chunk_type == "sources_found":
        source_list = "".join([f"<li>{src}</li>" for src in content])
        return f"<p>📚 Found {len(content)} potential sources:</p><ul>{source_list}</ul>"
    elif chunk_type == "scraping_start":
        return f"<p>🕸️ Starting to scrape {len(urls)} URLs...</p>"
    elif chunk_type == "scraping_end":
        return "<p>✅ Finished scraping.</p>"
    elif chunk_type == "processing_source":
        return f"<p>📄 Processing source: {url}</p>"
    elif chunk_type == "final_answer":
        sources_html = ""
        if sources:
            sources_html = "<h4>Sources Used:</h4><ul class='source-list'>"
            for i, s in enumerate(sources):
                title = s.get("title") or s.get("link", "Unknown source")
                link = s.get("link", "#")
                sources_html += f"<li class='source-item'><a href='{link}' target='_blank'>{title}</a></li>"
            sources_html += "</ul>"

        return f"""
            <div class="final-answer">
                <h3>Final Answer</h3>
                <div>{answer}</div>
                {sources_html}
            </div>
        """
    elif chunk_type == "error":
        error = chunk.get("error", "Unknown error")
        return f"<p>❌ <strong>Error:</strong> {error}</p>"
    elif chunk_type == "warning":
        warning = chunk.get("warning", "")
        return f"<p>⚠️ <strong>Warning:</strong> {warning}</p>"
    else:
        return f"<p>Unknown stream type: {chunk_type}</p>"


@app.post("/search")
async def search(request: Request):
    form = await request.form()
    query = form.get("query", "")

    if not query:
        return HTMLResponse("<p>Please enter a valid query</p>")

    async def generate_response():
        mode_description = "Deep Research Mode" if args.pro_mode else "Standard Search Mode"
        output_style = "Perplexity-Style" if args.pro_mode else "Standard"

        effective_max_sources = args.max_sources
        effective_max_sources_per_iteration = args.max_sources_per_iteration if args.pro_mode else 3

        yield "<div class='stream-output'>"
        yield f"<p><strong>Query:</strong> {query}</p>"

        if args.pro_mode:
            yield f"<p><em>Using {mode_description} | Output Style: {output_style} | Min Sources: {args.min_sources} | Max Sources: {effective_max_sources} | Max Sources Per Iteration: {effective_max_sources_per_iteration} | Max Iterations: {args.max_iterations}</em></p>"
        else:
            yield f"<p><em>Using {mode_description} | Output Style: {output_style} | Min Sources: {args.min_sources} | Max Sources: {effective_max_sources}</em></p>"

        try:

            async for chunk in search_tool.stream_forward(
                query=query,
                min_sources=args.min_sources,
                max_sources=effective_max_sources,
                pro_mode=args.pro_mode,
                max_iterations=args.max_iterations,
                max_sources_per_iteration=effective_max_sources_per_iteration,
            ):
                formatted_output = format_stream_chunk(chunk)
                print(f"[DEBUG] Processing chunk type: {chunk.get('type')}")
                if chunk.get("type") == "final_answer":
                    print(f"[DEBUG] Final answer: {chunk.get('answer')[:100]}...")
                yield formatted_output
                await asyncio.sleep(0.01)

        except Exception as e:
            import traceback

            traceback.print_exc()
            yield f"<p>❌ <strong>An error occurred:</strong> {str(e)}</p>"

        yield "</div>"

    return StreamingResponse(generate_response(), media_type="text/html")


if __name__ == "__main__":
    print("Starting OpenDeepSearch Web Demo Server...")
    print(f"Access the demo at http://localhost:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
