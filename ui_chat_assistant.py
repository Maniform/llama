# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama

import tkinter as tk
from tkinter import scrolledtext
import threading

def on_enter(event, send_button):
    send_button.invoke()

def process_send_message(input_entry, text_area, dialogs, generator, max_gen_len, temperature, top_p, thread):
    question = input_entry.get()
    if question:
        text_area.configure(state='normal')
        text_area.insert('end', "You : " + question + "\n\nAssistant :")
        text_area.configure(state='disabled')
        input_entry.delete(0, 'end')
        input_entry["state"] = "disabled"
        dialogs[0].append({"role": "user", "content": question})
        thread = threading.Thread(target=send_message, args=(input_entry, text_area, dialogs, generator, max_gen_len, temperature, top_p))
        thread.start()

def send_message(input_entry, text_area, dialogs, generator, max_gen_len, temperature, top_p):
    pos = text_area.index(tk.INSERT)
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        printFunction=update_text_area_text,
        printFunctionArguments=[text_area, pos],
    )
    dialogs[0].append({"role": "assistant", "content": results[-1]["generation"]["content"]})
    pos = text_area.index(tk.INSERT)
    text_area.after(0, lambda: text_area_insert(text_area, pos, "\n\n"))
    input_entry["state"] = "normal"

def update_text_area_text(generator, text, arguments):
    arguments[0].after(0, lambda: text_area_insert(arguments[0], arguments[1], text))

def text_area_insert(text_area, pos, text):
    text_area.configure(state='normal')
    text_area.delete(pos, tk.END)
    text_area.insert(pos, text)
    text_area.configure(state='disabled')
    text_area.see(tk.END)

def clear_text_area(text_area, dialogs):
    if dialogs[0][0]['role'] == "system":
        system_prompt = dialogs[0][0]['content']
        dialogs.clear()
        dialogs.extend([[{"role": "system", "content": system_prompt}]])
    else:
        dialogs.clear()
        dialogs.extend([[]])
    text_area.configure(state='normal')
    text_area.delete(1.0, tk.END)
    text_area.configure(state='disabled')

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = None
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [[{"role": "system", "content": "You are an expert helping answering questions."}]]

    thread = None

    root = tk.Tk()
    root.title("Assistant Llama")

    # Créer la zone de texte avec scrolling
    text_area = scrolledtext.ScrolledText(root, wrap='word', state='disabled')
    text_area.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

    # Créer la ligne d'entrée et le bouton "Envoyer"
    input_entry = tk.Entry(root)
    input_entry.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
    input_entry.bind('<Return>', lambda event: on_enter(event, send_button))

    send_button = tk.Button(root, text="Envoyer", command=lambda: process_send_message(input_entry, text_area, dialogs, generator, max_gen_len, temperature, top_p, thread))
    send_button.grid(row=1, column=1, padx=10, pady=10, sticky='e')

    clear_button = tk.Button(root, text="Effacer", command=lambda: clear_text_area(text_area, dialogs))
    clear_button.grid(row=1, column=2, padx=10, pady=10, sticky='e')

    # Redimensionner les cellules de la grille pour que la zone de texte ait la priorité
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)

    root.mainloop()

if __name__ == "__main__":
    fire.Fire(main)
