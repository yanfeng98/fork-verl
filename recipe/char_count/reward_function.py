from verl.utils.reward_score import math_reward

def char_count_reward_function(data_source, solution_str: str, ground_truth: str, extra_info=None) -> int:
    try:
        last_boxed_string: str = math_reward.last_boxed_only_string(solution_str)
        if last_boxed_string is None:
            return 0
        solution: str = math_reward.remove_boxed(last_boxed_string)
        if solution == ground_truth:
            return 1
        else:
            return 0
    except Exception:
        print(ground_truth, solution_str)
        return 0
