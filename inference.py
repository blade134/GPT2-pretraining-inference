from transformers import GPT2Tokenizer, GPT2LMHeadModel    #, set_seed
import torch

model_path = './LLM/out-12block-tomatoes-2'   # tomatoes-1
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

intput_text = "Unite States has 51 states,"

inputs = tokenizer(intput_text, return_tensors='pt')
print(inputs)

# input_ids_new = inputs.input_ids.to('cuda')

outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    # max_length=100,
    max_new_tokens=200,
    num_return_sequences=2,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,          # 开启采样
    top_k=50,               # top-k采样
    top_p=0.95,             # 核采样
    temperature=1.0,        # 温度
    no_repeat_ngram_size=2, # 避免重复n-gram
)

for i, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Sequence {i+1}: {generated_text}\n")