from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.data_utils import maybe_apply_chat_template

# 1. Load tokenizer and model
model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# 2. Load the Mixture-of-Thoughts dataset
dataset = load_dataset("open-r1/Mixture-of-Thoughts", 'all', split="train")
sample = dataset[0]  # First example

# 3. Print raw example
print("=== RAW EXAMPLE ===")
print(sample)

# 4. Apply chat template
processed = maybe_apply_chat_template(sample, tokenizer, chat_template="qwen2.5", add_generation_prompt=True)
print("\n=== PROMPT AFTER CHAT TEMPLATE ===")
print(processed["prompt"])
