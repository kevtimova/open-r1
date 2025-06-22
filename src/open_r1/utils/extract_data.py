import re



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
    raise ValueError(f"Programming language cpp or python not found in the prompt. {prompt}")
