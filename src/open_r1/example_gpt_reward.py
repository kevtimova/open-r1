
from openai import OpenAI
client = OpenAI()

prompt_template = """

You are RewardGPT. You output a reward given this input: question, reference answer, and prediction. The question, reference answer, and prediction will be wrapped in XML tags. If the prediction is correct, return "correct". If the prediction is incorrect, return "incorrect" wrapped in <reward>, such as <reward>correct</reward> or <reward>incorrect</reward>. Do not output anything else. Do not output any additional text or explanations. Just output the reward.

Additional background:
- The question is related to coding.
- The key part of the answer is the code itself, not the explanation.


<question>
{question}
</question>


<reference_answer>
{reference_answer}
</reference_answer>


<prediction>
{prediction}
</prediction>


Remember, you are RewardGPT. You output a reward given this input: question, reference answer, and prediction (shown above). If the prediction is correct, return "correct". If the prediction is incorrect, return "incorrect" wrapped in <reward>, such as <reward>correct</reward> or <reward>incorrect</reward>. Do not output anything else. Do not output any additional text or explanations. Just output the reward.
""".strip()


reference_free_prompt_template = """

You are RewardGPT. You output a reward given this input: question and prediction. The question and prediction will be wrapped in XML tags. If the prediction is correct, return "correct". If the prediction is incorrect, return "incorrect" wrapped in <reward>, such as <reward>correct</reward> or <reward>incorrect</reward>. Do not output anything else. Do not output any additional text or explanations. Just output the reward.

Additional background:
- The question is related to coding.
- The key part of the answer is the code itself, not the explanation.


<question>
{question}
</question>


<prediction>
{prediction}
</prediction>


Remember, you are RewardGPT. You output a reward given this input: question and prediction (shown above). If the prediction is correct, return "correct". If the prediction is incorrect, return "incorrect" wrapped in <reward>, such as <reward>correct</reward> or <reward>incorrect</reward>. Do not output anything else. Do not output any additional text or explanations. Just output the reward.
""".strip()

def generate_reward(question, reference_answer, prediction) -> int:
    prompt = prompt_template.format(
        question=question,
        reference_answer=reference_answer,
        prediction=prediction
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=20,
        stop=["</reward>"]
    )
    
    content = raw_content = response.choices[0].message.content

    if content.startswith("<reward>"):
        content = content[len("<reward>"):]
    if content not in ("correct", "incorrect"):
        print(f"WARNING: The response is not formatted correctly.\nResponse: {raw_content}")
        content = "incorrect"
    return 1 if content == "correct" else 0

def generate_reward_without_reference(question, prediction) -> int:
    prompt = reference_free_prompt_template.format(
        question=question,
        prediction=prediction
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=20,
        stop=["</reward>"]
    )
    
    content = raw_content = response.choices[0].message.content

    if content.startswith("<reward>"):
        content = content[len("<reward>"):]
    if content not in ("correct", "incorrect"):
        print(f"WARNING: The response is not formatted correctly.\nResponse: {raw_content}")
        content = "incorrect"
    return 1 if content == "correct" else 0

def example():
    question = "What is the capital of France?"
    reference_answer = "The capital of France is Paris."
    correct_prediction = "The capital of France is Paris."
    incorrect_prediction = "The capital of France is Rome."
    
    reward = generate_reward(question, reference_answer, correct_prediction)
    print(f"Correct Prediction Reward: {reward}")
    reward = generate_reward(question, reference_answer, incorrect_prediction)
    print(f"Incorrect Prediction Reward: {reward}")


def real_example(reference_free=False):
    import random
    from datasets import load_dataset

    if reference_free:
        def reward_func(question, reference_answer, prediction):
            print("Using reference-free reward generation.")
            return generate_reward_without_reference(question, prediction)
    else:
        reward_func = generate_reward

    print("Loading dataset...")

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("open-r1/Mixture-of-Thoughts", "code")

    # Randomly select a question.
    example = ds["train"][random.randint(0, len(ds["train"]) - 1)]
    question = example["messages"][0]["content"]
    reference_answer = example["messages"][-1]["content"]

    # import ipdb; ipdb.set_trace()

    # Randomly select a question and use its reference answer as a negative.
    incorrect_prediction = ds["train"][random.randint(0, len(ds["train"]) - 1)]["messages"][-1]["content"]

    print(f"Running reward generation (reference_free={reference_free})...")

    print(f"Question Word Count: {len(question.split())}")
    print(f"Reference Answer Word Count: {len(reference_answer.split())}")
    print(f"Incorrect Prediction Word Count: {len(incorrect_prediction.split())}")
    
    reward = reward_func(question, reference_answer, reference_answer)
    print(f"Correct Prediction Reward: {reward}")
    reward = reward_func(question, reference_answer, incorrect_prediction)
    print(f"Incorrect Prediction Reward: {reward}")


if __name__ == "__main__":
    # example()

    real_example(reference_free=False)
    real_example(reference_free=True)
