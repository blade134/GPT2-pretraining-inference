from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = './your/output/directory'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

input_text = "山东警方出警迅速，"

inputs = tokenizer(input_text, return_tensors='pt').to(device)
print(inputs)

outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=200,
    num_return_sequences=2,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
    no_repeat_ngram_size=2,
)

for i, output in enumerate(outputs):
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Sequence {i+1}: {generated_text}\n")
