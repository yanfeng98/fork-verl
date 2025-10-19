"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import re
import argparse
import datasets
from typing import Callable

def extract_solution(solution_str: str) -> str:
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution: str = solution.group(0)
    final_solution: str = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

"""
python examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path: str = args.local_dataset_path

    data_source: str = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset: datasets.DatasetDict = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset: datasets.DatasetDict = datasets.load_dataset(data_source, "main")

    train_dataset: datasets.Dataset = dataset["train"]
    test_dataset: datasets.Dataset = dataset["test"]

    instruction_following: str = 'Let\'s think step by step and output the final answer after "####".'

    def make_map_fn(split: str) -> Callable[[dict[str, str], int], dict[str, str|dict[str, int|str]|list[dict[str, str]]]]:
        def process_fn(example: dict[str, str], idx: int) -> dict[str, str|dict[str, int|str]|list[dict[str, str]]]:
            question_raw: str = example.pop("question")
            question: str = question_raw + " " + instruction_following

            answer_raw: str = example.pop("answer")
            solution: str = extract_solution(answer_raw)
            
            data: dict[str, str|dict[str, int|str]|list[dict[str, str]]] = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir: str = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
