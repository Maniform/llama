#!/bin/bash

script="example_chat_completion.py"
model="llama-2-7b-chat/"

set_model=false

for arg in "$@"; do
	if [ $set_model == true ]; then
		model="$arg"
		set_model=false
	elif [ $arg == "-model" ]; then
    	set_model=true
    else
    	script="$arg"
    fi
done

torchrun --nproc_per_node 1 $script --ckpt_dir $model --tokenizer_path tokenizer.model --max_seq_len 2048 --max_batch_size 4
