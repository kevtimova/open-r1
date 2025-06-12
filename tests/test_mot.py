import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.data_utils import maybe_apply_chat_template


def main():
    parser = argparse.ArgumentParser(description="Debug chat template application.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path (e.g. Qwen/Qwen2.5-7B)")
    parser.add_argument("--chat_template", type=str, default=None,
                        help="Optional chat template string to override tokenizer default")
    args = parser.parse_args()

    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if args.chat_template:
        tokenizer.chat_template = args.chat_template
        print("Overridden chat template.")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    # 2. Load Mixture-of-Thoughts dataset
    dataset = load_dataset("open-r1/Mixture-of-Thoughts", 'all', split="train")
    sample = dataset[0]

    # 3. Print raw example
    print("=== RAW EXAMPLE ===")
    print(sample)

    # 4. Apply chat template
    print("\n=== PROMPT AFTER CHAT TEMPLATE ===")
    try:
        import ipdb; ipdb.set_trace()  # Debugging breakpoint
        prompt = maybe_apply_chat_template(sample, tokenizer)
        print(prompt)
    except Exception as e:
        print("Failed to apply chat template:")
        print(e)


if __name__ == "__main__":
    main()