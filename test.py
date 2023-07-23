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

	dialogs = [[{"role": "system", "content": "You are an assitant."}]]
	user_input = ""
	while user_input != "exit":
		user_input = input(">>> ")
		if user_input != "exit":
			dialogs[0].append({"role": "user", "content": user_input})
		    generation_tokens, generation_logprobs = generator.generate(
				prompt_tokens=dialogs,
				max_gen_len=max_gen_len,
				temperature=temperature,
				top_p=top_p,
				logprobs=logprobs,
			)

		    #print(generator.tokenizer.decode(t) for t in generation_tokens)


if __name__ == "__main__":
    fire.Fire(main)