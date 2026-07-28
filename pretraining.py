from datasets import load_dataset
import torch

from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# --- config ---
model_name = None  # set to pretrained model name to continue training, else None to train from scratch
num_layers = 12
emb_size = 768
num_heads = 16
seq_length = 512
epochs = 1
batch_size = 4
output_dir = "./your/output/directory"

# --- tokenizer ---
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=seq_length)

# --- dataset (use a subset to keep example runnable) ---
# ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train')
ds = load_dataset('json', data_files='./data/2020-40_zh_head_0000.jsonl', split='train')

ds = ds.map(tokenize_batch, batched=True, remove_columns=ds.column_names)
ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
# print('ds.shape: ', ds.shape)

# --- model ---
if model_name:
    model = GPT2LMHeadModel.from_pretrained(model_name)
else:
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_embd=emb_size,
        n_layer=num_layers,
        n_head=num_heads,
        n_positions=seq_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

print(model)
print('总参数量:', sum(param.numel() for param in model.parameters()))

# --- set train args ---
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=200,
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
)
# print('train args:', training_args)

# --- data collator ---
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

print('-------------------start training-------------------')
trainer.train()

print('-------------------train fininshed-------------------')
print('-------------------可选：save model-------------------')
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
