#!/bin/bash

WANDB_PROJECT="PPIGPT" python run_clm.py \
    --model_name_or_path nferruz/ProtGPT2 \
    --train_file ../train.txt \
    --validation_file ../valid.txt \
    --tokenizer_name nferruz/ProtGPT2 \
    --do_train \
    --do_eval \
    --output_dir output2 \
    --learning_rate 1e-04 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --overwrite_output_dir \
    --num_train_epochs 15 \
    --block_size 1024 \
    --gradient_accumulation_steps 64 \
    --save_total_limit 2 \
    --save_strategy 'epoch'
