import argparse
from multiprocessing import Pool

import litellm
import pandas as pd
from tqdm import tqdm

input_prompt = """You are a precise answer extractor. Your job is to read a question and a detailed answer, then output ONLY the final answer without any explanation.

For example:
Question: "What is 2+2?"
Detailed Answer: "Let me calculate this. 2 plus 2 equals 4, which is a basic mathematical fact."
Final Answer: 4

Question: "What color is the sky on a clear day?"
Detailed Answer: "When we look up on a clear day, the sky appears blue due to a phenomenon called Rayleigh scattering."
Final Answer: blue

Question: "If my future wife has the same first name as the 15th first lady of the United States' mother and her surname is the same as the second assassinated president's mother's maiden name, what is my future wife's name?"
Detailed Answer: "The 15th First Lady of the United States was Ellen Wilson, and her mother's name was Hannah. The second assassinated president was Abraham Lincoln, and his mother's maiden name was Hodge. \n\nPutting that together, your future wife's name is **Hannah Hodge**."
Final Answer: Hannah Hodge

Now do this:
Question: {question}
Detailed Answer: {detailed_answer}
Final Answer:"""


def process_row(row):
    """Process a single row using litellm."""
    try:
        output = litellm.completion(
            model="openrouter/google/gemini-2.0-flash-001",
            messages=[
                {
                    "role": "user",
                    "content": input_prompt.format(question=row["question"], detailed_answer=row["original_answer"]),
                }
            ],
            temperature=0.3,
        )
        return output["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def process_dataframe(df, num_workers=4):
    """Process the entire dataframe using a pool of workers."""
    with Pool(num_workers) as pool:

        results = list(tqdm(pool.imap(process_row, [row for _, row in df.iterrows()]), total=len(df)))

    df["processed_output"] = results
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file using litellm in parallel")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (default: 4)")

    args = parser.parse_args()

    df = pd.read_json(args.input_file, lines=True)

    df = df.rename(columns={"answer": "original_answer"})

    processed_df = process_dataframe(df, num_workers=args.workers)
    processed_df = processed_df.rename(columns={"processed_output": "answer"})

    output_file = args.input_file.rsplit(".", 1)[0] + "_processed." + args.input_file.rsplit(".", 1)[1]
    processed_df.to_csv(output_file, index=False)
    print(f"Processed data saved to: {output_file}")
