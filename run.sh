#!/bin/bash

script="example_chat_completion.py"
model="llama-2-7b-chat/"
max_seq_len=1024

set_model=false
set_max_seq_len=false

for arg in "$@"; do
	if [ $arg == "-h" ]; then
		echo "Parameters :\n-model <model_dir>\n-max_seq_len <max_memory>"
		exit
	elif [ $set_model == true ]; then
		model="$arg"
		set_model=false
	elif [ $set_max_seq_len == true ]; then
		max_seq_len=$arg
		set_max_seq_len=false
	elif [ $arg == "-model" ]; then
    	set_model=true
    elif [ $arg == "-max_seq_len" ]; then
    	set_max_seq_len=true
    else
    	script="$arg"
    fi
done

torchrun --nproc_per_node 1 $script --ckpt_dir $model --tokenizer_path tokenizer.model --max_seq_len $max_seq_len --max_batch_size 4