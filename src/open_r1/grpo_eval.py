# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

# CUSTOM
import wandb
from transformers import Trainer
from trl.trainer.callbacks import _generate_completions, gather_object
from trl.trainer.callbacks import maybe_apply_chat_template, Optional, GenerationConfig
import pandas as pd

class LogCompletions:

    def __init__(
        self,
        trainer: Trainer,
        generation_config: Optional[GenerationConfig] = None,
        eval_dataset = None,
    ):
        self.trainer = trainer
        self.generation_config = generation_config
        self.table = []

        eval_dataset = eval_dataset or self.trainer.eval_dataset
        if eval_dataset is None:
            raise ValueError()
        self.eval_dataset = eval_dataset

    def run(self, batch_size, tokenizer):
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        prompts = self.eval_dataset["prompt"]
        prompts = [maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"] for prompt in prompts]
        completions = _generate_completions(
            prompts,
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            generation_config=self.generation_config,
            batch_size=batch_size,
        )
        # TODO: Add multi-gpu support.
        completions = gather_object(completions)
        prompts = gather_object(prompts)

        # Build the data to log
        data = list(zip(prompts, completions))
        self.table.extend(data)
        table = pd.DataFrame(columns=["prompt", "completion"], data=self.table)

        wandb.log({"completions": table})

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        wandb.init()
        init_wandb_training(training_args)

    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    DEMO = True

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = example[prompt_column][0]
        # completion = example[prompt_column][1] # noqa
        if DEMO:
            prompt['content'] = "... " + prompt['content'][-100:]
        return {"prompt": [prompt]}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] # TODO: Use actual eval set from local split.

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(eval_dataset)
        # trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)
        if trainer.accelerator.is_main_process:
            # self.vllm_client.generate(
            #                 prompts=ordered_set_of_prompts,
            #                 n=self.num_generations,
            #                 repetition_penalty=self.repetition_penalty,
            #                 temperature=self.temperature,
            #                 top_p=self.top_p,
            #                 top_k=-1 if self.top_k is None else self.top_k,
            #                 min_p=0.0 if self.min_p is None else self.min_p,
            #                 max_tokens=self.max_completion_length,
            #                 guided_decoding_regex=self.guided_decoding_regex,
            #             )
            generation_config = GenerationConfig(
                do_sample=True,
                repetition_penalty=training_args.repetition_penalty,
                temperature=training_args.temperature,
                max_tokens=training_args.max_completion_length,
            )
            if training_args.num_generations > 1:
                # Duplicate the eval dataset per generation.
                eval_dataset = eval_dataset.repeat(training_args.num_generations)

            log_completions = LogCompletions(trainer, generation_config=generation_config, eval_dataset=eval_dataset)
            log_completions.run(batch_size=training_args.per_device_eval_batch_size, tokenizer=tokenizer)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
