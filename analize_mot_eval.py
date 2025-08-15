"""
Analyze the results of the MOT eval.

"""

import argparse
import json
import re

def analyze_rewards(path="results.json"):
    with open(path, "r") as f:
        data = json.load(f)

    total_prompts = len(data)
    overall_rewards = []
    reward_at_k = []
    techniques = []

    print(f"Loaded {total_prompts} prompts")

    for item in data:
        # Get the completions and rewards
        completions = item.get("completions", [])
        rewards = [c.get("reward", 0.0) for c in completions]
        # extract technique, if present
        technique = [re.findall(r"<technique_name>(.*?)</technique_name>", c['completion'][0]['content'])[0] if re.findall(r"<technique_name>(.*?)</technique_name>", c['completion'][0]['content']) else 'NA' for c in completions]
        if not rewards:
            continue

        # Track the best, worst, and average reward
        best = max(rewards)
        reward_at_k.append(best)
        overall_rewards.extend(rewards)
        techniques.extend(technique)

    # Overall rewards statistics
    if overall_rewards:
        total = len(overall_rewards)
        global_avg = sum(overall_rewards) / total
        print("\nSummary:")
        print(f"  Total completions: {total}")
        print(f"  Global average reward: {global_avg:.2f}")
    
    # Reward at k
    reward_at_k_avg = sum(reward_at_k) / total_prompts
    print(f"  Reward at k: {reward_at_k_avg:.2f}")

    # Techniques distribution
    technique_counts = {t: techniques.count(t) for t in set(techniques)}
    print("\nTechniques distribution:")
    for technique, count in technique_counts.items():
        print(f"  {technique}: {count}")

if __name__ == "__main__":
    # Argparse for the path.
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="logs/results.json")
    args = parser.parse_args()

    # Analyze the rewards.
    analyze_rewards(args.path)