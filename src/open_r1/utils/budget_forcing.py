from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset


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
    truncated_dataset = Dataset.filter_mot_data(truncated_dataset)
    dataset['test'] = truncated_dataset
    # dataset.save_to_disk(f"{dataset}_truncated_{max_n_tokens}_tokens")

# Example usage and testing
if __name__ == "__main__":
    # Example MoT dataset sample
    sample_dataset = [
        {
            'prompt': 'Solve this math problem: 2 + 2 = ?',
            'completions': 'Let me think about this step by step. First, I need to understand what addition means. Addition is combining quantities. So when we have 2 + 2, we are combining two quantities of 2. This gives us 4. Therefore, 2 + 2 = 4.',
            'ground_truth': '4'
        },
        {
            'prompt': 'What is the capital of France?',
            'completions': [
                'The capital of France is Paris. Paris is located in the north-central part of France and is the most populous city in the country.',
                'France\'s capital city is Paris, which has been the political and cultural center of France for centuries.'
            ],
            'ground_truth': 'Paris'
        }
    ]
    
    # Truncate to 20 tokens max
    truncated_data = truncate_data(max_n_tokens=20, 
                                   dataset=sample_dataset,
                                   model_name="Qwen/Qwen2.5-7B-Instruct",
                                   final_answer_template="\n\nFinal answer: <answer>")
    
    # Print results
    for i, sample in enumerate(truncated_data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Original prompt: {sample['prompt']}")
        print(f"Truncated completion(s): {sample['completions']}")