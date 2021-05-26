# LRQA

(Name pending)

## Overview

This repository contains code to support experiments on long-document QA models. Currently it just works as a multiple-choice experiment repository.

### Tasks

LRQA supports either existing multiple-choice datasets (e.g. Cosmos QA, HellaSwag), or custom task datasets for quick iteration. In all cases, the goal is to map different datasets to a standard format of:

```
{
    "context": ...
    "query": ...
    "option_0" ...
      ..
    "option_N" ...
    "label": ...
}
```

Where the data will be formatted into inputs as:

```
[CLS] context [SEP] query option 
```
The strings are concatenated with *no spaces*. (Formatting may differ based on different tokenizers.)

Converters to this standard format are available for a set of multiple-choice tasks (see: `lrqa/tasks.py`):

- Cosmos QA


For custom tasks, you can provide a path to a folder containing files such as
```
config.json
train.jsonl
validation.jsonl
test.jsonl
```

The individual phases are optional. `config.json` specifies some configurations for the task, such as the number of choices. See `lrqa.tasks.CustomJSONLTask` for more details.

### Models

Models can be broken down into 2 categories:

**Encoder-based** models are based on `transformers.AutoModelForMultipleChoice`. All auto-modals should be compatible. These need to be fine-tuned, though tuned models can be applied across tasks.

**Generation-based** models are based on `transformers.AutoModelForCausalLM`. All auto-modals should be compatible. These can be run zero-shot, as the LM heads can be used directly to score continuations. 

Encoder-Decoder support is coming soon!

## Sample usage

#### Training + Evaluating an Encoder

```bash
EXPDIR=/path/to/experiment
python lrqa/run_lrqa.py \
    --model_name_or_path roberta-base \
    --model_mode mc \
    --task_name cosmosqa \
    --output_dir ${EXPDIR} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy no \
    --load_best_model_at_end \
    --num_train_epochs 1
```

#### Evaluating a Generator

```bash
EXPDIR=/path/to/experiment
python lrqa/run_lrqa.py \
    --model_name_or_path gpt2 \
    --model_mode mc \
    --task_name cosmosqa \
    --output_dir ${EXPDIR} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy no \
    --load_best_model_at_end \
    --num_train_epochs 1
```

#### Evaluating a Generator on a custom task

```bash
EXPDIR=/path/to/experiment
python lrqa/run_lrqa.py \
    --model_name_or_path gpt2 \
    --model_mode mc \
    --task_name custom \
    --task_base_path ./resources/example_jsonl_task \
    --output_dir ${EXPDIR} \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --save_strategy no \
    --load_best_model_at_end \
    --num_train_epochs 1
```
