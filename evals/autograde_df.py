import argparse
from multiprocessing import Pool, cpu_count

import litellm
import pandas as pd
from tqdm import tqdm

from evals.grader_prompts import GRADER_TEMPLATE


def grade_row(row_data):
    idx, row = row_data
    question = row["original_question"]
    predicted_answer = row["answer"]
    gold_answer = row["true_answer"]

    input_prompt = GRADER_TEMPLATE.format(question=question, predicted_answer=predicted_answer, target=gold_answer)

    try:
        output = litellm.completion(
            model="openrouter/google/gemini-2.0-flash-001",
            messages=[{"role": "user", "content": input_prompt}],
            temperature=0.0,
        )["choices"][0]["message"]["content"]
        return idx, output
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return idx, "Error"


def autograde_df(df_path, num_cpus=4):

    df = pd.read_json(df_path, lines=True)

    row_data = list(df.iterrows())

    n_processes = max(1, min(num_cpus, cpu_count()))
    print(f"Using {n_processes} processes")

    with Pool(n_processes) as pool:

        results = list(tqdm(pool.imap(grade_row, row_data), total=len(row_data), desc="Grading"))

    results.sort(key=lambda x: x[0])
    final_grades = [grade for _, grade in results]

    df["final_grade"] = final_grades

    df.to_json(df_path, orient="records", lines=True)
    print("Grading completed and results saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-grade answers in a DataFrame")
    parser.add_argument("df_path", type=str, help="Path to the DataFrame JSON file")
    parser.add_argument("--num_cpus", type=int, default=4, help="Number of CPU cores to use")

    args = parser.parse_args()
    autograde_df(args.df_path, args.num_cpus)
