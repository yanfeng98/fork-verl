import os
import random
import pandas as pd
from typing import Any
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

prompt_template: str = "How many {} are there in word {}?"


def generate_random_char() -> str:
    return chr(97 + random.randint(0, 25))


def create_prompt_response(min_length: int = 3, max_length: int = 5) -> tuple[str, str]:
    word_length: int = random.randint(min_length, max_length)
    target_count_number: int = random.randint(1, word_length)

    char_lst: list[str] = []
    target_char: str = generate_random_char()

    for _ in range(target_count_number):
        char_lst.append(target_char)

    for _ in range(word_length - target_count_number):
        while True:
            char: str = generate_random_char()
            if char != target_char:
                char_lst.append(char)
                break

    random.shuffle(char_lst)
    word: str = "-".join(char_lst)

    prompt: str = prompt_template.format(target_char, word)
    final_answer: list[str] = []

    number: int = 0
    for i, char in enumerate(char_lst):
        cot: str = f"{char}"
        if char != target_char:
            cot += " != "
        else:
            cot += " = "
            number += 1
        cot += f"{target_char}."

        final_answer.append(cot)

    conclusion: str = f"\\boxed{{{number}}} {target_char} in {word}."
    final_answer.append(conclusion)

    final_answer: str = "\n".join(final_answer)

    return prompt, final_answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_number", type=int, default=10000)
    parser.add_argument("--min_length", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--data_path", type=str, default="~/data/char_count")

    args: dict[str, Any] = vars(parser.parse_args())

    total_number: int = args["total_number"]
    min_length: int = args["min_length"]
    max_length: int = args["max_length"]
    data_path: int = args["data_path"]
    data_path: str = os.path.expanduser(data_path)

    full_output: list[tuple[str, str]] = []
    for _ in range(total_number):
        output: tuple[str, str] = create_prompt_response(min_length=min_length, max_length=max_length)
        full_output.append(output)

    random.shuffle(full_output)

    train_split_len: int = int(0.9 * len(full_output))
    train_outputs: list[tuple[int, int]] = full_output[:train_split_len]
    test_output: list[tuple[int, int]] = full_output[train_split_len:]

    sft_train_dataset: dict[str, list[str]] = {"prompt": [], "response": []}

    for o in train_outputs:
        sft_train_dataset["prompt"].append(o[0])
        sft_train_dataset["response"].append(o[1])

    sft_test_dataset: dict[str, list[str]] = {"prompt": [], "response": []}

    for o in test_output:
        sft_test_dataset["prompt"].append(o[0])
        sft_test_dataset["response"].append(o[1])

    sft_train_dataset: pd.DataFrame = pd.DataFrame(data=sft_train_dataset)
    sft_test_dataset: pd.DataFrame = pd.DataFrame(data=sft_test_dataset)

    folder: str = os.path.join(data_path, "sft")
    os.makedirs(folder, exist_ok=True)

    sft_train_dataset.to_parquet(os.path.join(folder, "train.parquet"))
    sft_test_dataset.to_parquet(os.path.join(folder, "test.parquet"))

    rl_train_dataset: dict[str, list[Any]] = {"prompt": [], "data_source": [], "ability": [], "reward_model": [], "extra_info": []}
    rl_test_dataset: dict[str, list[Any]] = {"prompt": [], "data_source": [], "ability": [], "reward_model": [], "extra_info": []}

    for o in train_outputs:
        prompt: str = o[0]
        response: str = o[1]
        prompt_with_template: list[dict[str, str]] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        rl_train_dataset["prompt"].append(prompt_with_template)
        rl_train_dataset["data_source"].append("char_count")
        rl_train_dataset["ability"].append("other")
        rl_train_dataset["reward_model"].append(
            {"style": "rule", "ground_truth": remove_boxed(last_boxed_only_string(response))}
        )
        rl_train_dataset["extra_info"].append({"response": response})

    for o in test_output:
        prompt: str = o[0]
        response: str = o[1]
        prompt_with_template: list[dict[str, str]] = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        rl_test_dataset["prompt"].append(prompt_with_template)
        rl_test_dataset["data_source"].append("char_count")
        rl_test_dataset["ability"].append("other")
        rl_test_dataset["reward_model"].append(
            {"style": "rule", "ground_truth": remove_boxed(last_boxed_only_string(response))}
        )
        rl_test_dataset["extra_info"].append({"response": response})

    rl_train_dataset: pd.DataFrame = pd.DataFrame(data=rl_train_dataset)
    rl_test_dataset: pd.DataFrame = pd.DataFrame(data=rl_test_dataset)

    folder: str = os.path.join(data_path, "rl")
    os.makedirs(folder, exist_ok=True)

    rl_train_dataset.to_parquet(os.path.join(folder, "train.parquet"))
    rl_test_dataset.to_parquet(os.path.join(folder, "test.parquet"))
