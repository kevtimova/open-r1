from datasets import load_dataset, DatasetDict
import argparse

def filter_dataset(dataset_name="open-r1/Mixture-of-Thoughts", domain='all', split="train"):
    """
    Load and filter the dataset based on the number of tokens.
    """
    # Load the full dataset
    ds = load_dataset(dataset_name, domain, split="train")

    # Prompt length
    for n in [10000, 5000, 3000]:
        # Filter out examples with more than n tokens
        filtered_ds = ds.filter(lambda x: x["num_tokens"] <= n)
        dataset = DatasetDict({"train": filtered_ds})

        # Save to local disk
        dataset.save_to_disk(f"data/mot_{domain}_filtered_{n//1000}k")

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Filter Mixture of Thoughts dataset based on token count.")
    parser.add_argument("--dataset_name", type=str, default="open-r1/Mixture-of-Thoughts", help="Name of the dataset to filter.")
    parser.add_argument("--domain", type=str, default="all", help="Domain: code|math|science|all.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to filter.")
    args = parser.parse_args()

    # Call the filter function with parsed arguments
    filter_dataset(dataset_name=args.dataset_name, domain=args.domain, split=args.split)
