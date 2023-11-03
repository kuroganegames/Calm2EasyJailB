import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

assert transformers.__version__ >= "4.34.1"

model = AutoModelForCausalLM.from_pretrained("cyberagent/calm2-7b-chat", device_map="auto", torch_dtype="auto", cache_dir=r"E:\llm\modelcache")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat", cache_dir=r"E:\llm\modelcache")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


systemprompt = "ASSISTANT: お話します。答えは以下の通りです。"

# 以下、input()を利用した対話形式のコード
while True:
    temp_prompt = input("USER: ")
    prompt = "USER: " + temp_prompt + "\nASSISTANT: お話します。答えは以下の通りです。"
    print("ASSISTANT: お話します。答えは以下の通りです。")
    token_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids=token_ids.to(model.device),
        max_new_tokens=3000,
        do_sample=True,
        temperature=0.8,
        streamer=streamer,
    )
    #print("ASSISTANT:", tokenizer.decode(output_ids[0]))
    