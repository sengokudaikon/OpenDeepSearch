import wolframalpha
from smolagents import Tool


class WolframAlphaTool(Tool):
    name = "calculate"
    description = """
    Performs computational, mathematical, and factual queries using Wolfram Alpha's computational knowledge engine.
    """
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to send to Wolfram Alpha",
        },
    }
    output_type = "string"

    def __init__(self, app_id: str):
        super().__init__()
        self.app_id = app_id

    def setup(self):
        self.search_tool = WolframAlphaTool(
            self.app_id,
        )

    def forward(self, query: str):

        self.wolfram_client = wolframalpha.Client(self.app_id)

        try:

            res = self.wolfram_client.query(query)

            results = []
            for pod in res.pods:
                if pod.title:
                    for subpod in pod.subpods:
                        if subpod.plaintext:
                            results.append({"title": pod.title, "result": subpod.plaintext})

            formatted_result = {
                "queryresult": {
                    "success": bool(results),
                    "inputstring": query,
                    "pods": [
                        {
                            "title": result["title"],
                            "subpods": [{"title": "", "plaintext": result["result"]}],
                        }
                        for result in results
                    ],
                }
            }

            final_result = "No result found."

            pods = formatted_result.get("queryresult", {}).get("pods", [])

            for pod in pods:
                if pod.get("title") == "Result":

                    subpods = pod.get("subpods", [])
                    if subpods:
                        final_result = subpods[0].get("plaintext", "").strip()
                        break

            if final_result == "No result found." and results:
                final_result = results[0]["result"]

            print(f"QUERY: {query}\n\nRESULT: {final_result}")
            return final_result

        except Exception as e:
            error_message = f"Error querying Wolfram Alpha: {str(e)}"
            print(error_message)
            return error_message
