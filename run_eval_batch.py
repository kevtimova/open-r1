import os
import asyncio
from typing import List, Tuple

from openai import AsyncOpenAI
from openai import APIStatusError, APITimeoutError, APIConnectionError
from tqdm import tqdm

import datasets
import argparse
import collections
import random
import re
import time
import json
from datetime import datetime
import numpy as np

from open_r1.rewards import code_reward

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_RETRIES = 4
PER_REQUEST_TIMEOUT_S = 60.0  # per attempt


def build_prompt(prompt: str, pick_strategy: str = "choose_best_fit"):
    techniques_list = ['DynamicProgramming', 'BruteForce', 'PrefixSum', 'Greedy', 'Hashing']
    additional_techniques = ['Stack', 'MathematicalCalculation', 'Simulation', 'Math']

    if pick_strategy == "choose_best_fit":
        random_technique = random.choice(techniques_list)
        parts = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', random_technique)
        if len(parts) == 1:
            joined = random_technique.upper()
        else:
            joined = ' '.join(parts)
        prompt_with_technique = f"""
Step 1: FIRST line must be JSON:
{{"technique":"{random_technique}"}}

Step 2: SECOND line must be an XML tag that repeats EXACTLY the same value:
<technique_name>{random_technique}</technique_name>

Important:
- The string inside <technique_name>…</technique_name> must be IDENTICAL to the "technique" field in the JSON.
- Do NOT invent your own tag name (<{random_technique}>…</{random_technique}>).
- Do NOT write "ChosenAlgorithm" literally (<technique_name>ChosenAlgorithm</technique_name>).
- Do NOT change spacing or casing (<technique_name>{joined}</technique_name>).

Correct example:
{{"technique":"{random_technique}"}}
<technique_name>{random_technique}</technique_name>

Incorrect example:
{{"technique":"{random_technique}"}}
<{random_technique}>ChosenAlgorithm</{random_technique}>

{prompt}

Remember to specify the chosen technique for your solution in the format: <technique_name>ChosenAlgorithm</technique_name>. 
"""
    elif pick_strategy == "choose_randomly_from_list":
        random_technique, chosen_technique = random.sample(techniques_list, 2)
        # 50% of the time choose a technique from another list
        if random.random() < 0.5:
            chosen_technique = random.choice(additional_techniques)
        parts = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', random_technique)
        if len(parts) == 1:
            joined = random_technique.upper()
        else:
            joined = ' '.join(parts)
        prompt_with_technique = f"""
Step 1: FIRST line must be JSON:
{{"technique":"{random_technique}"}}

Step 2: SECOND line must be an XML tag that repeats EXACTLY the same value:
<technique_name>{random_technique}</technique_name>

Important:
- The string inside <technique_name>…</technique_name> must be IDENTICAL to the "technique" field in the JSON.
- Do NOT invent your own tag name (<{random_technique}>…</{random_technique}>).
- Do NOT write "ChosenAlgorithm" literally (<technique_name>ChosenAlgorithm</technique_name>).
- Do NOT change spacing or casing (<technique_name>{joined}</technique_name>).

Correct example:
{{"technique":"{random_technique}"}}
<technique_name>{random_technique}</technique_name>

Incorrect example:
{{"technique":"{random_technique}"}}
<{random_technique}>ChosenAlgorithm</{random_technique}>

For this problem it has been preselected to use technique {chosen_technique}. Use {chosen_technique} and proceed normally. Still mention technique {chosen_technique} within the expected format.

{prompt}

Remember to specify the chosen technique for your solution in the format: <technique_name>ChosenAlgorithm</technique_name>. 
"""
    return prompt_with_technique


def build_messages(prompt: str):
    return [
        {"role": "system", "content": "Be concise and accurate."},
        {"role": "user", "content": prompt},
    ]


async def call_once(client: AsyncOpenAI, messages, verification):
    # Minimal retry on transient errors (429/5xx/timeout/connection)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                timeout=PER_REQUEST_TIMEOUT_S,
            )
            completion = resp.choices[0].message.content
            return completion, verification
        except (APIStatusError, APITimeoutError, APIConnectionError) as e:
            if attempt == MAX_RETRIES:
                return f"[ERROR] {type(e).__name__}: {e}", verification
            # simple exponential backoff
            await asyncio.sleep(0.5 * (2 ** (attempt - 1)))


async def worker(idx: int, prompt: str, verification: dict, client: AsyncOpenAI, sem: asyncio.Semaphore) -> Tuple[int, str]:
    async with sem:
        completion, verification = await call_once(client, build_messages(prompt), verification)
    return idx, prompt, completion, verification


async def run_batch(prompts: List[str], verification: List[dict], concurrency: int = 16) -> List[Tuple[int, str, dict]]:
    client = AsyncOpenAI()  # uses OPENAI_API_KEY from env
    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(worker(i, p, v, client, sem)) for i, (p, v) in enumerate(zip(prompts, verification))]

    results: List[Tuple[int, str]] = [None] * len(tasks)  # type: ignore
    with tqdm(total=len(tasks), desc="Calling ChatGPT") as pbar:
        for fut in asyncio.as_completed(tasks):
            idx, prompt, completion, verification = await fut
            results[idx] = (idx, prompt, completion, verification)
            pbar.update(1)
    return results

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


# ---- example usage ----
if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset. If not provided, will use the online version.")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1)
    parser.add_argument("--solution_strategy", type=str, default="choose_randomly_from_list")
    parser.add_argument("--num_solutions", type=int, default=4)
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "digitalocean"])
    args = parser.parse_args()

    # Dataset
    if args.data_path:
        dataset = datasets.load_from_disk(args.data_path)
    else:
        dataset = datasets.load_dataset("open-r1/Mixture-of-Thoughts", "code")
    subset = [item for i, item in enumerate(dataset['train']) if i>=args.start_index and i < args.end_index]
    print(f"Loaded {len(subset)} samples indexed [{args.start_index}:{args.end_index}] from the full {len(dataset['train'])} dataset.")

    skipped = collections.Counter()
    prompts = []
    verification_info = []
    # Generate the prompts along with test cases
    for example in subset:
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
        test_cases = extract_test_cases(prompt)
        if len(test_cases) == 0:
            skipped["no_test_cases"] += 1
            continue

        # Example Format:
        # {'test_cases': [{'type': 'stdin_stdout', 'input': '2\n4 1\n5 5 5 5\n3 2\n0 0 0\n', 'output': '10\n0\n'}], 'language': 'python'}
        this_verification_info = {"test_cases": test_cases, "language": language}

        for i in range(args.num_solutions):
            prompt_with_technique = build_prompt(prompt, pick_strategy=args.solution_strategy)
            prompts.append(prompt_with_technique)
            verification_info.append(this_verification_info)
    
    print(f"Skipped so far: {skipped}")

    # Generate responses
    out = asyncio.run(run_batch(prompts, verification_info, concurrency=16))
    # out[i] == (i, "response for prompt i")
    print(f"Collected {len(out)} results.")

    # Obtain rewards
    num_batches = len(out) // args.batch_size
    results = []
    grouped = collections.defaultdict(list)
    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M")
    for i in range(num_batches + 1):
        # Get the batch.
        batch = out[i * args.batch_size:(i + 1) * args.batch_size]
        if not batch:
            continue
        prompts = [item[1] for item in batch]
        completions = [[{"content": item[2]}] for item in batch]
        verification_info = [item[3] for item in batch]
        # Evaluate the completions.
        rewards = code_reward(completions, num_parallel=args.batch_size, provider_type="morph", verification_info=verification_info)
        print(rewards)
        results.extend(rewards)
        # Combine rewards by prompt.
        for prompt, completion, reward in zip(prompts, completions, rewards):
            grouped[prompt].append({"completion": completion, "reward": reward})
        # Save results to JSON file
        json_data = [{"prompt": prompt, "completions": completions} for prompt, completions in grouped.items()]
        output_path = f"logs/results_{args.solution_strategy}_{args.num_solutions}_{args.start_index}-{args.end_index}_{timestamp}.json"
        with open(output_path, "w") as f:
            json.dump(json_data, f)

    results = np.array(results)
    print(results.mean())
    print(results.std())
    print(results.min())
    print(results.max())

