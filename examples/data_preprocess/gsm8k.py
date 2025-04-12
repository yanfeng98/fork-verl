"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import argparse


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


# python examples/data_preprocess/gsm8k.py --local_dir ./data/gsm8k --max_samples 36 --val_size 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/gsm8k')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--val_size', type=float, default=0.1)
    args = parser.parse_args()

    data_source = 'openai/gsm8k'
    dataset = datasets.load_dataset(data_source, 'main')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            question = question_raw + ' ' + instruction_following
            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)

            data = {
                "data_source": data_source,
                "ability": "math",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "question": question_raw,
                    'answer': answer_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir: str = args.local_dir
    max_samples: int = args.max_samples
    val_size: float|int = args.val_size

    if max_samples is not None:
        max_samples = min(max_samples, len(train_dataset))
        dataset = train_dataset.select(range(max_samples))

        val_size = int(val_size) if val_size > 1 else val_size
        dataset = dataset.train_test_split(test_size=val_size, seed=42)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
