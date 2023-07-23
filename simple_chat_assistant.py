# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
     
    question = ""
    index = 0
    dialogs = [[{"role": "system", "content": "You are an expert helping answering questions."}]]
    with open("log.txt", "w") as log:
        while question != "exit":
            question = input(">>> ")
            if question != "exit":
                dialogs[index].append({"role": "user", "content": question})
                log.write("You: " + question + "\n\n")
                log.flush()

                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                print("\n" + results[-1]["generation"]["content"] + "\n")
                log.write("Assistant:" + results[-1]["generation"]["content"] + "\n\n")
                log.flush()
                dialogs[index].append({"role": "assistant", "content": results[-1]["generation"]["content"]})

if __name__ == "__main__":
    fire.Fire(main)
