import numpy as np
import os
import torch
from typing import Dict, Optional
from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    LEDConfig,
    LEDForConditionalGeneration,
    LEDTokenizerFast,
)
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.data.data_collator import default_data_collator

import lrqa.tasks as tasks
from lrqa.utils.hf_utils import parse_args, last_checkpoint_handling
from lrqa.utils.io_utils import write_json, show_json


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    padding_strategy: PaddingStrategy = field(
        default="max_length",
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    parallelize: bool = field(
        default=False,
        metadata={
            "help": "Whether to parallelize the model."
        }
    )
    truncation_strategy: TruncationStrategy = field(
        default="only_second",
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    torch_dtype_fp16: bool = field(
        default=False,
        metadata={"help": "Enable this and set model_revision='fp16' for fp16 GPT-J"},
    )
    eval_phase: str = field(
        default="validation",
        metadata={"help": "Phase for evaluation (train|validation|test)"},
    )
    predict_phases: str = field(
        default="test",
        metadata={"help": "Comma separated phases for evaluation (train|validation|test)"},
    )

    def __post_init__(self):
        self.padding_strategy = PaddingStrategy(self.padding_strategy)
        self.truncation_strategy = TruncationStrategy(self.truncation_strategy)


def tokenize_examples_for_led(examples, tokenizer, max_seq_length: int,
                              padding_strategy: PaddingStrategy,
                              truncation_strategy: TruncationStrategy):
    option_keys = sorted([
        key for key in examples
        if key.startswith("option_")
    ])
    input_strs = []
    input2_strs = []
    target_strs = []
    for i in range(len(examples[option_keys[0]])):
        all_options = "\n".join(
            [f"Choice {j + 1}: {examples[option_key][i].strip()}" for j, option_key in enumerate(option_keys)])
        input_str = f"{all_options}\n\nQuestion: {examples['query'][i].strip()}"
        input2_str = f"Context: {examples['context'][i].strip()}"
        target_str = f"{examples['label'][i] + 1}"
        input_strs.append(input_str)
        input2_strs.append(input2_str)
        target_strs.append(target_str)

    tokenized_inputs = tokenizer(
        input_strs,
        input2_strs,
        max_length=max_seq_length,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
    )
    tokenized_targets = tokenizer(
        target_strs,
        max_length=1,
        padding=padding_strategy,
        truncation=truncation_strategy,
        return_tensors="pt",
        add_special_tokens=False,
    )
    target_ids = tokenized_targets["input_ids"]
    target_ids[target_ids[:, :] == tokenizer.pad_token_id] = -100

    input_ids = tokenized_inputs["input_ids"].numpy()
    is_eos = input_ids == tokenizer.eos_token_id
    input2_index = (is_eos[:, :-1] & is_eos[:, 1:]).argmax(-1)
    assert (input2_index > 0).all()
    global_attention_mask = np.zeros_like(input_ids, dtype=np.int64)
    for i, index in enumerate(input2_index):
        global_attention_mask[i, :index] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_inputs["attention_mask"].numpy(),
        "label": target_ids.numpy(),
        "global_attention_mask": global_attention_mask,
        # "input2_index": input2_index,
    }


def get_tokenized_dataset(task: tasks.Task, dataset_dict,
                          tokenizer,
                          max_seq_length: int,
                          padding_strategy: PaddingStrategy,
                          truncation_strategy: TruncationStrategy,
                          ) -> Dict:
    tokenized_dataset = {}
    for phase in ["train", "validation", "test"]:
        if phase not in dataset_dict:
            continue
        standard_examples = dataset_dict[phase].map(
            task.standardize_examples,
            batched=True,
            remove_columns=task.drop_columns,
        )
        tokenize_examples = lambda examples: tokenize_examples_for_led(
            examples, tokenizer, max_seq_length, padding_strategy, truncation_strategy)
        tokenized_examples = standard_examples.map(tokenize_examples, batched=True)
        tokenized_dataset[phase] = tokenized_examples
    return tokenized_dataset


class LEDTaskTrainer(Trainer):
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys,
    ):
        batch_size, seq_len = inputs["labels"].shape
        preds = torch.ones_like(inputs["labels"]) * -100
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        for i in range(batch_size):
            out = self.model.generate(
                input_ids=inputs["input_ids"][i:i+1],
                attention_mask=inputs["attention_mask"][i:i+1],
                global_attention_mask=inputs["global_attention_mask"][i:i+1],
            )
            # generate outputs </s> first
            # preds[i, :out.shape[1]-1] = out[0][1:]
            length = min(seq_len, len(out[0])-1)
            # preds[i, :length] = out[0][1:length+1]
            # For base
            preds[i, :length] = out[0][2:length + 2]
        return None, preds, inputs["labels"]


def main():
    model_args, task_args, training_args = parse_args(HfArgumentParser((
        ModelArguments,
        tasks.TaskArguments,
        TrainingArguments,
    )))
    config = LEDConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    tokenizer = LEDTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    model = LEDForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
    )
    if model_args.parallelize:
        model.parallelize()
    else:
        model = model.cuda()

    task = tasks.get_task(task_args=task_args)
    dataset_dict = task.get_datasets()
    tokenized_dataset_dict = get_tokenized_dataset(
        task=task,
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        max_seq_length=model_args.max_seq_length,
        padding_strategy=model_args.padding_strategy,
        truncation_strategy=model_args.truncation_strategy,
    )

    def clean_decode(input_ids):
        return tokenizer.decode([x for x in input_ids if x != -100])

    def compute_metrics(eval_preds):
        num_examples = eval_preds.predictions.shape[0]
        num_correct = 0
        clean_preds = []
        clean_labels = []
        for i in range(num_examples):
            clean_pred = clean_decode(eval_preds.predictions[i])
            clean_label = clean_decode(eval_preds.label_ids[i])
            if clean_pred == clean_label:
                num_correct += 1
            clean_preds.append(clean_pred)
            clean_labels.append(clean_label)
        return {"accuracy": num_correct / num_examples}

    trainer = LEDTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_dict.get("train"),
        eval_dataset=tokenized_dataset_dict.get("validation"),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    checkpoint = last_checkpoint_handling(training_args=training_args, model_args=model_args)
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model(output_dir=os.path.join(training_args.output_dir, "checkpoint-last"))
        # noinspection PyArgumentList
        trainer.log_metrics("train", train_result.metrics)
        # noinspection PyArgumentList
        trainer.save_metrics("train", train_result.metrics)
        # noinspection PyArgumentList
        trainer.save_state()

    if training_args.do_eval:
        validation_metrics = trainer.evaluate(eval_dataset=tokenized_dataset_dict[model_args.eval_phase])
        write_json(validation_metrics, os.path.join(training_args.output_dir, f"{model_args.eval_phase}_metrics.json"))
        show_json(validation_metrics)

    if training_args.do_predict:
        for phase in model_args.predict_phases.split(","):
            predictions = trainer.predict(test_dataset=tokenized_dataset_dict[phase]).predictions
            torch.save(predictions, os.path.join(training_args.output_dir, f"{phase}_predictions.p"))


if __name__ == "__main__":
    main()
