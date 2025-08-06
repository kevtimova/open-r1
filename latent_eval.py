from datasets import load_dataset
from openai import OpenAI
import openai
import time
import os
import requests
import json
import re

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def generate_solution(prompt, sketch, api_provider="digitalocean", num_retries=5):
    """
    Generate a solution based on the prompt and sketch.
    """
    if api_provider.lower() == "digitalocean":
        # Generate response from DigitalOcean
        MODEL_ACCESS_KEY = os.getenv("DIGITALOCEAN_API_KEY")
        url = "https://inference.do-ai.run/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MODEL_ACCESS_KEY}",
            "Content-Type": "application/json"
        }
        propt_with_sketch = f"{prompt}\n\nSketch solution: {sketch}"
        payload = {
            "model": "llama3.3-70b-instruct",
            "messages": [
                {"role": "user", "content": propt_with_sketch}
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
            import ipdb; ipdb.set_trace()
            solution = generate_solution(prompt, sketch)
