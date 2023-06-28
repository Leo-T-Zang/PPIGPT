#!/bin/bash

declare -A groups
groups[0]="1e-3,4,15"
groups[1]="2e-3,2,15"
groups[2]="1e-3,2,25"

for index in ${!groups[@]}; do
    IFS=',' read learning_rate gradient_accumulation_steps num_epochs <<< "${groups[$index]}"

    # Convert learning_rate to string suitable for folder names (replace '.' with '_')
    learning_rate_str=${learning_rate//./_}

    # Create unique output directory based on parameters
    output_dir="output_lr_${learning_rate_str}_g_${gradient_accumulation_steps}_e_${num_epochs}"

    echo "Starting training for Group: $((index+1)) with learning_rate: $learning_rate, gradient_accumulation_steps: $gradient_accumulation_steps, and num_epochs: $num_epochs"

    WANDB_PROJECT="PPIGPT" python run_clm.py \
        --model_name_or_path nferruz/ProtGPT2 \
        --train_file ../train.txt \
        --validation_file ../valid.txt \
        --tokenizer_name nferruz/ProtGPT2 \
        --do_train \
        --do_eval \
        --output_dir $output_dir \
        --learning_rate $learning_rate \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --overwrite_output_dir \
        --num_train_epochs $num_epochs \
        --block_size 1024 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_total_limit 2 \
        --save_strategy 'epoch'

    echo "Finished training for Group: $((index+1)). Output saved to $output_dir"
done
