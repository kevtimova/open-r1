from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset, DatasetDict
import argparse

def truncate_data(max_n_tokens, 
                  dataset_path, 
                  model_name, 
                  final_answer_template):
    """
    Truncate the MoT dataset completions to a maximum number of tokens and append final answer template.
    
    Args:
        max_n_tokens: Maximum number of tokens to keep from each completion
        dataset_path: Path to the dataset directory
        model_name: Model name for tokenizer (default: "gpt-4")
        final_answer_template: Template to append after truncation
        
    Returns:
        None (saves the truncated dataset to disk)
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the original dataset (don't modify this)
    original_dataset = load_from_disk(dataset_path)
    test_set = original_dataset['test']

    # Generate the truncated dataset    
    truncated_data = {}
    truncated_data['messages'] = []

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
        truncated_data['messages'].append([
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': truncated_completion}
        ])
    
    # Copy other fields from original test set
    truncated_data['num_tokens'] = test_set['num_tokens']  # Keep the original num_tokens field
    truncated_data['source'] = test_set['source']  # Keep the original source field

    # Create new dataset with truncated test set
    truncated_test_dataset = Dataset.from_dict(truncated_data)
    
    # Create a new DatasetDict that copies the original structure but replaces the test set
    new_dataset = DatasetDict()
    
    # Copy all splits from original dataset
    for split_name in original_dataset.keys():
        if split_name == 'test':
            new_dataset[split_name] = truncated_test_dataset
        else:
            new_dataset[split_name] = original_dataset[split_name]
    
    # Save new truncated dataset using the original path
    output_path = f"{dataset_path}_truncated_{max_n_tokens}_tokens"
    new_dataset.save_to_disk(output_path)
    print(f"Truncated dataset saved to: {output_path}")

# Example usage and testing
if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Truncate MoT dataset completions.")
    parser.add_argument("--dataset", type=str, default="data/mot_code_python_filtered_5K")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_n_tokens", type=int, default=20)
    parser.add_argument("--final_answer_template", type=str, default="\n\nFinal answer: <answer>")
    args = parser.parse_args()
    
    # Truncate to specified tokens max
    truncate_data(max_n_tokens=args.max_n_tokens, 
                  dataset_path=args.dataset, 
                  model_name=args.model_name, 
                  final_answer_template=args.final_answer_template)
    
