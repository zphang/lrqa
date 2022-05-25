# QuALITY training code (LRQA)

This LRQA repository contains code for running baselines on the QuALITY data. 

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
The strings are concatenated with *no spaces*. Hence, it is recommended to prepend spaces to option and query values. (Formatting may differ based on different tokenizers.)

Converters to this standard format are available for a set of multiple-choice tasks (see: `lrqa/tasks.py`):

- Cosmos QA


For custom tasks, you can provide a path to a folder containing files such as
```
config.json
train.jsonl
validation.jsonl
test.jsonl
```

The individual phases are optional. Each phase is contained in a single `jsonl` file, with the keys as specified above. `config.json` specifies some configurations for the task, such as the number of choices. See `lrqa.tasks.CustomJSONLTask` for more details. See `./resources/example_jsonl_task` for an example. 

### Models

Models can be broken down into 2 categories:

**Encoder-based** models are based on `transformers.AutoModelForMultipleChoice`. All auto-modals should be compatible. These need to be fine-tuned, though tuned models can be applied across tasks.

Note: Hugging Face's implementation runs `.forward` on all options at once, which can increases ram consumption. Adjust batch sizes and gradient accumulation steps accordingly.

**Generation-based** models are based on `transformers.AutoModelForCausalLM`. All auto-modals should be compatible. These can be run zero-shot, as the LM heads can be used directly to score continuations. 

Encoder-Decoder support is coming soon!

## Usage setup

It is recommended to add the path to this repository to your `PYTHONPATH`, e.g.

```bash
export PYTHONPATH=/path/to/lrqa/:${PYTHONPATH} 
```

Install requirements as necessary from `requirements.txt`. It is also recommended though not necessary to run code from within this folder.

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
    --model_mode generation \
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
    --model_mode generation \
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


## Extractive Preprocessing

Use the HTML-cleaned version of the QuALITY data. You can run the preprocessing for generating the extractive baseline inputs as follows:

```bash
python lrqa/scripts/extraction.py \
    --input_base_path /path/to/input/data \
    --output_base_path /path/to/output/data \
    --scorer rouge \
    --query_type question
```

This will prepare the data in a format that is suitable for the above `run_lrqa` scripts.

## Applications

### Applying to BBQ

Here is some sample code for running evaluation on [BBQ](https://arxiv.org/abs/2110.08193).

First, we preprocess (just writing to standardized JSONL format) and fine-tune on RACE.

```bash
EXP_FOL=...
HF_MODEL_NAME=roberta-base
BATCH_SIZE=8
GRAD_ACCUM_STEPS=4
python lrqa/scripts/race_preproc.py \
    --data_path ${EXP_FOL}/race
python lrqa/run_lrqa.py \
    --model_name_or_path ${HF_MODEL_NAME} \
    --model_mode mc \
    --max_seq_length 512 \
    --task_name custom \
    --task_base_path ${EXP_FOL}/race \
    --output_dir ${EXP_FOL}/race_run \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --warmup_ratio 0.1 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --do_train --do_eval --do_predict --predict_phases validation
```

Next we preprocess all the BBQ data (cloned from https://github.com/nyu-mll/BBQ), and run evaluation on each of the categories using fine-tuned models.
```bash
BBQ_DATA=...  # clone BBQ repo and point to `data` folder 
python lrqa/scripts/bbq_preproc.py \
    --input_data_path=${BBQ_DATA} \
    --data_path ${EXP_FOL}/bbq
for CATEGORY in Age Disability_status Gender_identity Nationality Physical_appearance Race_ethnicity Race_x_SES Race_x_gender Religion SES Sexual_orientation; do
    python lrqa/run_lrqa.py \
        --model_name_or_path ${EXP_FOL}/race_run/checkpoint-last \
        --model_mode mc \
        --max_seq_length 512 \
        --task_name custom \
        --task_base_path ${EXP_FOL}/bbq/${CATEGORY}/ \
        --output_dir ${EXP_FOL}/bbq_runs/${CATEGORY}/ \
        --per_device_eval_batch_size ${BATCH_SIZE} \
        --do_eval --do_predict --predict_phases validation
done
```

The predictions should be stored in a PyTorch pickle at `${EXP_FOL}/bbq_runs/${CATEGORY}/validation_predictions.p`.
