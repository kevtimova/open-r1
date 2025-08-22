from datasets import load_dataset
from openai import OpenAI
import openai
import time
import os
import requests
import json
import re
import random

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List of solution strategies
solution_strategies = """Dijkstra’s Algorithm, Bellman–Ford Algorithm, Floyd–Warshall Algorithm, A* Search Algorithm, Kruskal’s Algorithm, Prim’s Algorithm, Tarjan’s Algorithm, Kosaraju’s Algorithm, Union-Find, Disjoint Set Union, Union by Rank, Path Compression, Graph Traversal DFS, Graph Traversal BFS, Topological Sorting, Flood Fill, Heavy-Light Decomposition, Lowest Common Ancestor Binary Lifting, Huffman Coding, Fractional Knapsack, Coin Change, Job Sequencing with Deadlines, Egyptian Fraction Representation, Interval Scheduling, Merge Sort, Quick Sort, Heap Sort, Counting Sort, Radix Sort, Binary Search, Binary Search on Answer, Meet in the Middle, Knuth Optimization, Divide and Conquer DP Optimization, Bitmask Dynamic Programming, Segment Tree, Fenwick Tree, Binary Indexed Tree, Trie, Knuth–Morris–Pratt Algorithm, Rabin–Karp Algorithm, Z-function, Suffix Arrays, Manacher’s Algorithm, Convex Hull Graham Scan, Convex Hull Andrew’s Monotone Chain, Sweep Line Algorithm, Line Sweep for Geometry Problems, GCD, Modular Arithmetic, Prime Factorization, Fast Fourier Transform, Coordinate Compression, Mo’s Algorithm, Brute Force, Sliding Window, Two Pointers"""

# Generate solution strategy for each example
def generate_solution_strategy(example, api_provider="digitalocean", num_sketches=4, num_retries=5):
    problem = example["messages"][0]["content"]
    latent_prompt = f"""
    You will be given a coding problem. Your task is to generate multiple sketches of solutions.

    Guidelines:
    - A sketch is just a few sentences explaining what a solution could look like.
    - Use your best judgement on what is considered more or less complex. A less complex might be brute force. Recursive or DP could be considered more complex. In general, more complex solutions are harder to implement and require more creativity to come up with. Each sketch should have a short title.
    - Also include a rationale explaining how this solution is more complex than the previous.
    - Each sketch has an ID starting from 1. 1, 2, 3, etc.
    - The output format is: <output><id>...</id><title>...</title><sketch>...</sketch><rationale>...</rationale></output>

    Coding problem:
    {problem}

    Now generate {num_sketches} sketches in increasing complexity following the described output format. Only output the sketches, no other text.
    """

    if api_provider.lower() == "openai":
        # Generate response from OpenAI
        response = None
        for i in range(num_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": latent_prompt}],
                )
                break
            except openai.RateLimitError as e:
                print(f"Rate limit error: {e}")
                if i < num_retries - 1:  # Don't sleep on the last attempt
                    sleep_time = 10 * (2 ** i)  # Exponential backoff
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        if response is None:
            raise RuntimeError(f"Failed to get a response after {num_retries} retries.")
        content = response.choices[0].message.content
        
    elif api_provider.lower() == "digitalocean":
        # Generate response from DigitalOcean
        MODEL_ACCESS_KEY = os.getenv("DIGITALOCEAN_API_KEY")

        url = "https://inference.do-ai.run/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MODEL_ACCESS_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3.3-70b-instruct",
            "messages": [
                {"role": "user", "content": latent_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }

        response = None
        for i in range(num_retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    break
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    if i < num_retries - 1:  # Don't sleep on the last attempt
                        sleep_time = 10 * (2 ** i)  # Exponential backoff
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
            except requests.RequestException as e:
                print(f"Request error: {e}")
                if i < num_retries - 1:  # Don't sleep on the last attempt
                    sleep_time = 10 * (2 ** i)  # Exponential backoff
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        
        if response is None or response.status_code != 200:
            raise RuntimeError(f"Failed to get a successful response from DigitalOcean API after {num_retries} retries.")
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}. Use 'openai' or 'digitalocean'")
    
    return content

def extract_sketches(content):
    """
    Extract the sketches from the content.
    <output><id>...</id><title>...</title><sketch>...</sketch><rationale>...</rationale></output>
    """
    matches = re.findall(r"<sketch>(.*?)</sketch>", content, re.DOTALL)
    return matches

def generate_solution(prompt, 
                      sketch=None, 
                      api_provider="digitalocean", 
                      pick_strategy="choose_best_fit",
                      num_retries=5):
    """
    Generate a solution based on the prompt and sketch.
    """
    techniques_list = ['DynamicProgramming', 'BruteForce', 'PrefixSum', 'Greedy', 'Hashing', 'TwoPointers', 'GraphTraversal', 'SegmentedArray', 'BinarySearch']
    if sketch is not None:
        prompt = f"{prompt}\n\nUse the following sketch solution to generate a full solution: {sketch}. Think before you write your code."
    elif pick_strategy == "choose_best_fit":
        random_technique = random.choice(techniques_list)
        parts = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', random_technique)
        if len(parts) == 1:
            joined = random_technique.upper()
        else:
            joined = ' '.join(parts)
        prompt = f"""
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
        parts = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', random_technique)
        if len(parts) == 1:
            joined = random_technique.upper()
        else:
            joined = ' '.join(parts)
        prompt = f"""
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
    # Get a response from OpenAI API
    if api_provider.lower() == "openai":
        # Generate response from OpenAI
        response = None
        for i in range(num_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except openai.RateLimitError as e:
                print(f"Rate limit error: {e}")
                if i < num_retries - 1:  # Don't sleep on the last attempt
                    sleep_time = 10 * (2 ** i)  # Exponential backoff
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        if response is None:
            raise RuntimeError(f"Failed to get a response after {num_retries} retries.")
        content = response.choices[0].message.content

    # Get a response from DigitalOcean
    if api_provider.lower() == "digitalocean":
        # Generate response from DigitalOcean
        MODEL_ACCESS_KEY = os.getenv("DIGITALOCEAN_API_KEY")
        url = "https://inference.do-ai.run/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MODEL_ACCESS_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama3.3-70b-instruct",
            # "model": "DeepSeek-R1-distill-llama-70B",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 10000
        }

        response = None
        for i in range(num_retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    break
                else:
                    print(f"Error {response.status_code}: {response.text}")
                    if i < num_retries - 1:  # Don't sleep on the last attempt
                        sleep_time = 10 * (2 ** i)  # Exponential backoff
                        print(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
            except requests.RequestException as e:
                print(f"Request error: {e}")
                if i < num_retries - 1:  # Don't sleep on the last attempt
                    sleep_time = 10 * (2 ** i)  # Exponential backoff
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        
        if response is None or response.status_code != 200:
            raise RuntimeError(f"Failed to get a successful response from DigitalOcean API after {num_retries} retries.")
    return content

if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("open-r1/Mixture-of-Thoughts", "code", split="train")
    # Example usage with DigitalOcean
    provider = "digitalocean"

    for i in range(10):
        # Generate solution strategies
        content_digitalocean = generate_solution_strategy(dataset[i], api_provider=provider)
        # Extract the sketches
        sketches = extract_sketches(content_digitalocean)
        # Prompt for the solution
        prompt = dataset[i]["messages"][0]["content"]

        for sketch in sketches:
            # Generate a solution based on the sketch
            solution = generate_solution(prompt, sketch)
