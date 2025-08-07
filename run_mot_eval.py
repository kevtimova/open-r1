"""
Use code execution rewards to evaluate the ground truth MoT code data.

TODO: Support two ways of unit tests: 1. Extract from prompt, 2. Download from codeforces.

"""

# Code extraction utils

import re

def extract_test_cases(prompt: str) -> list[dict]:
    """Extract test cases from the prompt."""
    input_blocks = re.findall(r'```input\s*\n(.*?)\n```', prompt, re.DOTALL)
    output_blocks = re.findall(r'```output\s*\n(.*?)\n```', prompt, re.DOTALL)
    # Ensure that input and output blocks are paired correctly
    if len(input_blocks) != len(output_blocks):
        print("Warning: Mismatched input and output blocks found in the prompt.")
        return []
    # If no input or output blocks are found, return empty lists
    elif len(input_blocks) == 0:
        print("Warning: No input or output blocks found in the prompt.")
        return []
    # Same format expected by code_reward.
    # Add newline to the end of each input and output block, if needed.
    input_blocks = [input_block + "\n" if not input_block.endswith("\n") else input_block for input_block in input_blocks]
    output_blocks = [output_block + "\n" if not output_block.endswith("\n") else output_block for output_block in output_blocks]
    test_cases = [{"type": "stdin_stdout", "input": input_block, "output": output_block} for input_block, output_block in zip(input_blocks, output_blocks)]
    return test_cases

def extract_programming_language(prompt: str) -> str:
    """Extract programming language from the prompt."""
    match = re.findall(r'```(\w+)', prompt)
    match = [word.lower() for word in match]
    if "cpp" in match or "c++" in match:
        return "cpp"
    elif "python" in match:
        return "python"
    raise ValueError(f"Unknown programming language: {prompt}")

def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer

# MAIN

import collections
import json
import datasets
import numpy as np # noqa
from tqdm import tqdm
from open_r1.rewards import code_reward
from latent_eval import extract_sketches, generate_solution_strategy, generate_solution

dataset = datasets.load_dataset("open-r1/Mixture-of-Thoughts", "code")

def batch_iterator(batch_size=10,
                   use_type_1_tests=True,
                   use_type_2_tests=False,
                   solution_strategy='ground_truth',
                   num_solutions=4):
    skipped = collections.Counter()
    completions, verification_info, prompts = [], [], []

    for i in range(len(dataset["train"])):
        # Obtain the training sample.
        example = dataset["train"][i]
        
        # Extract the prompt.
        prompt = example["messages"][0]["content"]

        # Get language.
        language = extract_programming_language(prompt)
        assert language in ("python", "cpp")
        if language != "python":
            skipped["non_python"] += 1
            continue

        # Extract the ground truth generation and make sure it's a valid code snippet.
        ground_truth_generation = example["messages"][-1]["content"]
        code_snippet = extract_code(ground_truth_generation, language=language)
        if code_snippet == "":
            skipped["no_code_snippet"] += 1
            continue

        # Unit Tests Type 1: Extract from prompt.
        if use_type_1_tests:
            test_cases = extract_test_cases(prompt)
            if len(test_cases) == 0:
                skipped["no_test_cases"] += 1
                continue

        # Unit Tests Type 2: Download from codeforces.
        if use_type_2_tests:
            raise NotImplementedError("Type 2 tests not implemented yet.")

        # Example Format:
        # {'test_cases': [{'type': 'stdin_stdout', 'input': '2\n4 1\n5 5 5 5\n3 2\n0 0 0\n', 'output': '10\n0\n'}], 'language': 'python'}
        this_verification_info = {"test_cases": test_cases, "language": language}

        # Extract the solution for a valid sample.
        if solution_strategy == 'use_latent':
            solution_strategies = generate_solution_strategy(example, num_sketches=num_solutions)
            sketches = extract_sketches(solution_strategies)
            for sketch in sketches:
                # Add solution to batch.
                generation = generate_solution(prompt, sketch)
                completion = [{"content": generation}]
                completions.append(completion)
                # Add test cases to batch.
                verification_info.append(this_verification_info)
                # Add prompt to batch.
                prompts.append(prompt)
        elif solution_strategy == 'no_latent':
            for i in range(num_solutions):
                # Add solution to batch.
                generation = generate_solution(prompt)
                completion = [{"content": generation}]
                completions.append(completion)
                # Add test cases to batch.
                verification_info.append(this_verification_info)
                # Add prompt to batch.
                prompts.append(prompt)
        elif solution_strategy == 'ground_truth':
            # Add solution to batch.
            completion = [{"content": ground_truth_generation}]
            completions.append(completion)
            # Add test cases to batch.
            verification_info.append(this_verification_info)
            # Add prompt to batch.
            prompts.append(prompt)
        else:
            raise ValueError(f"Unknown solution strategy: {solution_strategy}")

        # Yield the batch if it's full. 
        if len(completions) >= batch_size:
            print(f"Skipped so far: {skipped}")
            yield completions, verification_info, prompts
            completions, verification_info, prompts = [], [], []

    # Yield the last batch.
    if len(completions) > 0:
        print(f"Skipped so far: {skipped}")
        yield completions, verification_info, prompts

# # Canary: Does code execution actually work as expected?
# language = "python"
# code_snippet = """
# ```python
# def main():
#     user_input = input()
#     values = list(map(int, user_input.split()))
#     print(sum(values))

# if __name__ == "__main__":
#     main()
# ```
# """.strip()

# test_cases = [{"type": "stdin_stdout", "input": "1 2 3\n", "output": "6\n"}, {"type": "stdin_stdout", "input": "1 2 3\n", "output": "5\n"}]
# verification_info = [{"test_cases": [test_cases[0]], "language": language}, {"test_cases": [test_cases[1]], "language": language}]
# rewards = code_reward([[{"content": code_snippet}], [{"content": code_snippet}]], provider_type="morph", verification_info=verification_info)
# expected_rewards = [1.0, 0.0]
# assert rewards == expected_rewards, f"Rewards: {rewards} != Expected rewards: {expected_rewards}"
# print("Canary passed.")

# Evaluation: Evaluate the MoT data.
# TODO: Add batching.
# TODO: Add patch code?
batch_size = 10
num_batches = len(dataset["train"]) // batch_size
results = []
grouped = collections.defaultdict(list)
strategy = "use_latent"
for completions, verification_info, prompts in tqdm(batch_iterator(batch_size, solution_strategy=strategy), total=num_batches):
    rewards = code_reward(completions, provider_type="morph", verification_info=verification_info)
    print(rewards)
    results.extend(rewards)
    # Combine rewards by prompt.
    for prompt, completion, reward in zip(prompts, completions, rewards):
        grouped[prompt].append({"completion": completion, "reward": reward})
    json_data = [{"prompt": prompt, "completions": completions} for prompt, completions in grouped.items()]
    with open(f"results_{strategy}.json", "w") as f:
        json.dump(json_data, f)

results = np.array(results)
print(results.mean())
print(results.std())
print(results.min())
print(results.max())

