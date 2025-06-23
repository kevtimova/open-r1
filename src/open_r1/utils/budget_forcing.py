from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
import argparse

def truncate_data(max_n_tokens, 
                  dataset, 
                  model_name, 
                  final_answer_template):
    """
    Truncate the MoT dataset completions to a maximum number of tokens and append final answer template.
    
    Args:
        max_n_tokens: Maximum number of tokens to keep from each completion
        dataset: List of MoT dataset samples, each containing 'completions' field
        model_name: Model name for tokenizer (default: "gpt-4")
        final_answer_template: Template to append after truncation
        
    Returns:
        List of modified samples with truncated completions
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the dataset 
    dataset = load_from_disk(dataset) if isinstance(dataset, str) else dataset
    test_set = dataset['test']

    # Generate the truncated dataset    
    truncated_dataset = {}
    truncated_dataset['messages'] = []

    for text in test_set['messages']:
        prompt = text[0]['content']
        completion = text[1]['content']

        # Tokenize the completions
        tokenized_completions = tokenizer(completion, return_tensors='pt', 
                                          truncation=True, max_length=max_n_tokens)

        # Decode the truncated completion
        truncated_completion = tokenizer.decode(tokenized_completions['input_ids'][0],
                                                skip_special_tokens=True)
        
        # Check if the truncated completion has <answer> tag and remove anything after it
        if "<answer>" in truncated_completion:
            truncated_completion = truncated_completion.split("<answer>")[0].strip()
            print("Warning: <answer> tag found in completion. Truncated completion will not include it and anything after it.")

        # Append final answer template
        if final_answer_template:
            truncated_completion += final_answer_template

        # Store the truncated sample
        truncated_dataset['messages'].append([
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': truncated_completion}
        ])

    # Save new truncated dataset 
    truncated_dataset = Dataset.from_dict(truncated_dataset)
    dataset['test'] = truncated_dataset
    dataset.save_to_disk(f"{dataset}_truncated_{max_n_tokens}_tokens")

# Example usage and testing
if __name__ == "__main__":
    # Arguments
    argparse = argparse.ArgumentParser(description="Truncate MoT dataset completions.")
    argparse.add_argument("--dataset", type=str, default="data/mot_code_python_filtered_5K")
    argparse.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    argparse.add_argument("--max_n_tokens", type=int, default=20)
    argparse.add_argument("--final_answer_template", type=str, default="\n\nFinal answer: <answer>")
    args = argparse.parse_args()
    
    # Truncate to 20 tokens max
    import ipdb; ipdb.set_trace()
    truncated_data = truncate_data(max_n_tokens=args.max_n_tokens, 
                                   dataset=args.dataset, 
                                   model_name=args.model_name, 
                                   final_answer_template=args.final_answer_template)
    
    # Print results
    for i, sample in enumerate(truncated_data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Original prompt: {sample['prompt']}")
        print(f"Truncated completion(s): {sample['completions']}")