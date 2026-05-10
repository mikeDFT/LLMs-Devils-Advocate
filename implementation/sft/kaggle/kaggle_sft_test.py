import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from kaggle_secrets import UserSecretsClient
from transformers import TrainerCallback
from huggingface_hub import login

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(hf_token)

MAX_SEQ_LENGTH = 2048
BASE_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
DATASET_REPO = "MikeDFT/devils-advocate-sft"
ADAPTER_REPO = "MikeDFT/devils-advocate-adapter-test"

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
# disable thinking mode
tokenizer.chat_template = tokenizer.chat_template.replace(
    "enable_thinking=True", "enable_thinking=False"
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
)

print("Loading and formatting dataset...")
ds = load_dataset(DATASET_REPO, split="train")
ds = ds.select(range(15))

tokenizer = get_chat_template(tokenizer, chat_template="chatml")

SYSTEM_PROMPT = (
    "You are the Devil's Advocate. Your sole purpose is to oppose, attack, and dismantle "
    "every claim the user makes. Expose weak evidence, logical fallacies, and flawed premises. "
    "Never concede, agree, or validate. Never say 'good point', 'you're right', or anything sycophantic. "
    "Cite specific names, dates, and data to destroy their position."
)

def format_examples_batched(batch):
    texts = []
    for premise, response, context, fallacy in zip(batch["user_premise"], batch["chosen_response"], batch["rag_context"], batch["fallacy_analysis"],):
        user_message = (
            f"Context Information:\n{context}\n\n"
            f"Fallacy Analysis:\n{fallacy}\n\n"
            f"User Argument:\n{premise}"
        )
        convo = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response}
        ]
        texts.append(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

ds = ds.map(format_examples_batched, batched=True, remove_columns=ds.column_names)
ds = ds.train_test_split(test_size=0.05, seed=1230946767)

training_args = SFTConfig(
    output_dir="outputs",
    max_seq_length=MAX_SEQ_LENGTH,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=2,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=25,
    save_steps=500,
    save_total_limit=2,
    warmup_ratio=0.05,
    optim="adamw_8bit",
    seed=1230946767,
    report_to="none", 
    eval_strategy="steps",
    eval_steps=100,
    dataset_text_field="text",
    packing=False,
    padding_free=False,
)

class FlushingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step} | Loss: {logs.get('loss', 'N/A')}", flush=True)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    processing_class=tokenizer,
    callbacks=[FlushingCallback()],
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
sample = next(iter(trainer.get_train_dataloader()))
assert (sample["labels"] != -100).any(), "All labels are masked"
n_total = sample["labels"].numel()
n_active = (sample["labels"] != -100).sum().item()
print(f"Label check passed — {n_active}/{n_total} tokens unmasked ({100*n_active/n_total:.1f}%)")

print("Starting training...")
last_ckpt = None
if os.path.isdir("outputs"):
    ckpts = [d for d in os.listdir("outputs") if d.startswith("checkpoint-")]
    if ckpts:
        last_ckpt = os.path.join("outputs", sorted(ckpts, key=lambda x: int(x.split("-")[1]))[-1])
        print(f"Resuming from {last_ckpt}")
trainer.train(resume_from_checkpoint=last_ckpt)

print("Training complete! Uploading adapter to Hugging Face...")
model.push_to_hub(ADAPTER_REPO, token=hf_token)
tokenizer.push_to_hub(ADAPTER_REPO, token=hf_token)
print("Upload successful. You can now close Kaggle!")
