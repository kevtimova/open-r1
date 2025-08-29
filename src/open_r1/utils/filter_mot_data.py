from datasets import load_dataset, Dataset
import argparse
from open_r1.utils import extract_programming_language, extract_test_cases, extract_code


def filter_dataset(dataset_name="open-r1/Mixture-of-Thoughts",
                   domain='all',
                   split="train",
                   split_sizes=[10000, 8000, 5000, 3000],
                   language=None,
                   test_size=10,
                   test_repeat=1,
                   seed=42):
    """
    Load and filter the dataset based on the number of tokens.
    """
    # Load the full dataset
    ds = load_dataset(dataset_name, domain, split=split)
    print(f"Loaded {len(ds)} samples from the {dataset_name} dataset, domain={domain}, split={split}.")

    # Remove non-python or cpp code
    ds = ds.filter(lambda x: extract_programming_language(x["messages"][0]["content"]) in ["python", "cpp"])
    print(f"Filtered non-python/cpp code. Current length: {len(ds)}")

    # Filter the dataset by programming language if specified
    if language is not None:
        ds = ds.filter(lambda x: extract_programming_language(x["messages"][0]["content"]) == language)
        print(f"Filtered for programming language: {language}. Current length: {len(ds)}")

    # Remove samples without a code snippet
    ds = ds.filter(lambda x: extract_code(x["messages"][-1]["content"], language=extract_programming_language(x["messages"][0]["content"])) != "")
    print(f"Filtered samples without code snippets. Current length: {len(ds)}")

    # Remove samples without test cases
    def has_test_cases(example):
        prompt = example["messages"][0]["content"]
        test_cases = extract_test_cases(prompt)
        return len(test_cases) > 0
    ds = ds.filter(lambda x: has_test_cases(x))
    print(f"Filtered samples without test cases. Current length: {len(ds)}")

    # Prompt length
    for n in split_sizes:
        # Filter out examples with more than n tokens
        filtered_ds = ds.filter(lambda x: x["num_tokens"] <= n)
        print(f"Filtered for max {n} tokens. Current length: {len(filtered_ds)}.")

        # Randomly split the dataset into train and test sets
        filtered_ds = filtered_ds.train_test_split(test_size=test_size, seed=seed)

        # Repeat test set sevearal times to get more samples
        if test_repeat > 1:
            new_test = {}
            for key in filtered_ds["test"].features.keys():
                new_test[key] = filtered_ds["test"][key] * test_repeat
            filtered_ds["test"] = Dataset.from_dict(new_test)

        # Save to local disk
        if language is None:
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
    parser.add_argument("--test_size", type=int, default=10, help="Number of test samples in split.")
    parser.add_argument("--test_repeat", type=int, default=1, help="Number of times to repeat the test set.")
    parser.add_argument("--language", type=str, default=None, help="Programming language to filter by (default: all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Call the filter function with parsed arguments
    split_sizes = list(map(int, args.split_sizes.split(',')))
    filter_dataset(dataset_name=args.dataset_name,
                   domain=args.domain,
                   language=args.language,
                   split=args.split,
                   split_sizes=split_sizes,
                   test_size=args.test_size,
                   test_repeat=args.test_repeat,
                   seed=args.seed)
    
