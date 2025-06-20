import re


def extract_test_cases(prompt: str) -> list[dict]:
        """Extract test cases from the prompt."""
        input_blocks = re.findall(r'```input\s*\n(.*?)\n```', prompt, re.DOTALL)
        output_blocks = re.findall(r'```output\s*\n(.*?)\n```', prompt, re.DOTALL)
        # Ensure that input and output blocks are paired correctly
        if len(input_blocks) != len(output_blocks):
            return [""], [""]
        # If no input or output blocks are found, return empty lists
        elif len(input_blocks) == 0:
            return [""], [""]
        # If there is only one input and output block, return them as lists
        elif len(input_blocks) == 1:
            inputs = input_blocks[0].splitlines()
            outputs = output_blocks[0].splitlines()
            # If the number of inputs and outputs is not the same
            if len(inputs) != len(outputs):
                # If the number of inputs is divisible by the number of outputs, group them                
                if len(inputs) % len(outputs) == 0:
                    group_size = len(inputs) // len(outputs)
                    in_items = []
                    out_items = []
                    for i in range(len(outputs)):
                        in_items.append("\n".join(inputs[i * group_size:(i + 1) * group_size]))
                        out_items.append(outputs[i])
                else:
                    # Otherwise, return empty lists
                    return [""], [" "]
            else:
                # If the number of inputs and outputs is the same, return them as lists
                return inputs, outputs
        else:
            return input_blocks, output_blocks
        
def extract_programming_language(prompt: str) -> str:
    """Extract programming language from the prompt."""
    match = re.findall(r'```(\w+)', prompt)
    match = match.map(lambda x: x.lower())
    if "cpp" in match or "c++" in match:
        return "cpp"
    elif "python" in match:
        return "python"
    return "python"  # Default to Python if no language is found