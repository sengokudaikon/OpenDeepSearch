import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()


class WebSearchEvaluator:
    def __init__(self, model: str, output_path: Path, num_workers: int = 4, trial: int = 0):
        self.model = model
        self.output_path = output_path
        self.num_workers = num_workers
        self.trial = trial

        self.processed_questions = set()
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        self.processed_questions.add(result["question"])
                    except Exception as e:
                        print(f"Error processing line: {line.strip()}, Exception: {e}")
                        continue

    def worker_init(self):
        """Initialize OpenAI client for each worker."""

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))

    def evaluate_single(self, row: pd.Series) -> Dict[str, Any]:
        """Evaluate a single question with its true answer."""

        if row["question"] in self.processed_questions:
            return None

        if not hasattr(self, "client"):
            self.worker_init()

        try:
            start_time = time.time()
            response = self.client.responses.create(
                model=self.model, tools=[{"type": "web_search_preview"}], input=row["question"]
            )
            end_time = time.time()
            result = {
                "question": row["question"],
                "true_answer": row["true_answer"],
                "answer": response.output_text,
                "model": self.model,
                "time_taken": end_time - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            return result
        except Exception as e:
            return {
                "question": row["question"],
                "true_answer": row["true_answer"],
                "answer": None,
                "error": str(e),
                "model": self.model,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    def save_result(self, result: Dict[str, Any]) -> None:
        """Save a single result to the JSONL file."""
        with open(self.output_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def evaluate_batch(self, df: pd.DataFrame) -> None:
        """Evaluate questions in parallel using multiple workers."""
        with ProcessPoolExecutor(max_workers=self.num_workers, initializer=self.worker_init) as executor:

            rows = [row for _, row in df.iterrows()]

            with tqdm(total=len(rows), desc="Processing questions") as pbar:

                futures = [executor.submit(self.evaluate_single, row) for row in rows]

                for future in futures:
                    result = future.result()
                    if result is not None:
                        self.save_result(result)
                    pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate questions using GPT-4 with web search")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save results (default: output)",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        default="./evals/datasets/frames_test_set.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("LITELLM_EVAL_MODEL_ID", os.getenv("LITELLM_MODEL_ID", "gpt-4o-mini")),
        help="Model to use for evaluation",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--trial", type=int, default=0, help="Trial number for this evaluation run (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"evaluation_results_{args.model}_trial{args.trial}.jsonl"

    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    evaluator = WebSearchEvaluator(
        model=args.model, output_path=output_path, num_workers=args.num_workers, trial=args.trial
    )

    print(f"Starting evaluation with model {args.model} using {args.num_workers} workers...")
    evaluator.evaluate_batch(df)
    print(f"Results saved to {output_path}")

    results_df = pd.read_json(output_path, lines=True)
    print("\nResults summary:")
    print(f"Model: {args.model}")
    print(f"Total evaluations: {len(results_df)}")
    print(f"Successful evaluations: {len(results_df[~results_df['answer'].isna()])}")
    print(f"Failed evaluations: {len(results_df[results_df['answer'].isna()])}")


if __name__ == "__main__":
    main()
