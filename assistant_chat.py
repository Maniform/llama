# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import curses

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
    dialogs = [[]]
    cur = curses.initscr()
    curses.curs_set(1)
    while question != "exit":
        y, x = cur.getyx()
        cur.addstr(y, x, ">>> ")
        y, x = cur.getyx()
        question = cur.getstr(y, x).decode('latin-1')
        # question = input(">>> ")
        if question != "exit":
            dialogs[index].append({"role": "user", "content": question})

            y, x = cur.getyx()
            cur.addstr(y, x, "\n")

            results = generator.chat_completion(
                cur,
                dialogs,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            # print("\n" + results[-1]["generation"]["content"] + "\n")
            y, x = cur.getyx()
            cur.addstr(y, x, "\n\n")
            #cur.addstr(y, x, "\n" + results[-1]["generation"]["content"] + "\n")
            dialogs[index].append({"role": "assistant", "content": results[-1]["generation"]["content"]})

if __name__ == "__main__":
    fire.Fire(main)
