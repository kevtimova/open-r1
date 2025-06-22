from datasets import load_dataset, DatasetDict
import argparse
from open_r1.utils import extract_programming_language

def filter_dataset(dataset_name="open-r1/Mixture-of-Thoughts",
                   domain='all',
                   split="train",
                   split_sizes=[10000, 8000, 5000, 3000],
                   language="all",
                   test_size=0.1,
                   seed=42):
    """
    Load and filter the dataset based on the number of tokens.
    """
    # Load the full dataset
    ds = load_dataset(dataset_name, domain, split=split)
    if language != "all":
        # Filter the dataset by programming language if specified
        ds = ds.filter(lambda x: extract_programming_language(x["messages"][0]["content"]) == language)

    # Prompt length
    for n in split_sizes:
        # Filter out examples with more than n tokens
        filtered_ds = ds.filter(lambda x: x["num_tokens"] <= n)

        # Randomly split the dataset into train and test sets
        filtered_ds = filtered_ds.train_test_split(test_size=test_size, seed=seed)

        # Save to local disk
        if language == "all":
            filtered_ds.save_to_disk(f"data/mot_{domain}_filtered_{n//1000}k")
        else:
            filtered_ds.save_to_disk(f"data/mot_{domain}_{language}_filtered_{n//1000}k")
        

if __name__ == "__main__":
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Filter Mixture of Thoughts dataset based on token count.")
    parser.add_argument("--dataset_name", type=str, default="open-r1/Mixture-of-Thoughts", help="Name of the dataset to filter.")
    parser.add_argument("--domain", type=str, default="all", help="Domain: code|math|science|all.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to filter.")
    parser.add_argument("--split_sizes", type=str, default="10000,8000,5000,3000", help="Comma-separate list of number of tokens to filter by.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--language", type=str, default="all", help="Programming language to filter by (default: all).")
    args = parser.parse_args()

    # Call the filter function with parsed arguments
    split_sizes = list(map(int, args.split_sizes.split(',')))
    filter_dataset(dataset_name=args.dataset_name,
                   domain=args.domain,
                   split=args.split,
                   split_sizes=split_sizes,
                   test_size=args.test_size,
                   seed=args.seed)
    
