import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)

from peft import LoraConfig
from trl import SFTTrainer


from random import randint
from time import sleep
import logging
from datetime import datetime
from pathlib import Path

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


log_file_name = "xgen_fineTune_{:%Y-%m-%d-%H-%M}.log".format(now)

logging.basicConfig(
    filename=log_file_name,
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(lineno)04d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

logging.debug("xgen------ xgen fine tune log file ")
logging.debug("xgen------ Set device ")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

logging.debug("xgen------ Device = " + device)

DO_TEST = False
if DO_TEST:
    logging.debug("xgen------ Load pretrained tokenizer xgen-7b-8k-base")

    tokenizer = AutoTokenizer.from_pretrained(
        "Salesforce/xgen-7b-8k-base", trust_remote_code=True
    )

    logging.debug("xgen------ Load pretrained model xgen-7b-8k-base")
    model = AutoModelForCausalLM.from_pretrained(
        "Salesforce/xgen-7b-8k-base", torch_dtype=torch.bfloat16
    )

    model = model.to(device)

    logging.debug("xgen------ Generate text")

    inputs = tokenizer("DataCamp is one he ...", return_tensors="pt").to(device)

    sample = model.generate(**inputs, max_length=128)

    logging.debug(tokenizer.decode(sample[0]))
    # take off gpu
    del model
    del tokenizer

training_epochs = 50

# Model from HF
base_model = "Salesforce/xgen-7b-8k-base"

# Fine-tuned model
new_model = "xgen-7b-8k-tuned_" + str(training_epochs) + "epochs"

logging.debug("xgen------ Set up dataset ")

import pandas as pd

sentiment_train_df = pd.read_csv("data/finBERT_TRAIN_TEST_VALIDATE/train.csv", sep="\t")
sentiment_train_df = sentiment_train_df[["text", "label"]]

system_prompt = "[INST] Classify the text into neutral, negative, or positive. Text: "
num_samples = len(sentiment_train_df)
df = pd.DataFrame(columns=["text"])
for i, data in sentiment_train_df.iterrows():
    if i > num_samples:
        break
    prompt = system_prompt + data["text"] + "[/INST] " + data["label"]
    print(prompt)
    df.loc[i] = prompt


logging.debug("xgen------ Set up dataset ")

from datasets import Dataset

dataset = Dataset.from_pandas(df)


compute_dtype = torch.float16

logging.debug("xgen------ Set up bitandbytes ")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype
)

logging.debug("xgen------ Set up model from_pretrained")
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=quant_config, device_map="auto"
)

logging.debug("xgen------ Set up tokenizer ")

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

logging.debug("xgen------ Set up lora config ")
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

logging.debug("xgen------ Set up training args ")
training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=training_epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

logging.debug("xgen------ Set up trainer ")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
logging.debug("xgen------ Training... ")
trainer.train()

prompt = "Classify the text into neutral, negative, or positive. Text: Comparable operating profit for the quarter decreased from EUR510m while sales increased from EUR860m , as compared to the third quarter 2007"

pipe = pipeline(
    task="text-generation", model=model, tokenizer=tokenizer, max_length=200
)

result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]["generated_text"])

trainer.model.save_pretrained(new_model)

# See https://github.com/huggingface/transformers/issues/28472
# We had to use transformer 4.33 to avoid the issue with saving
trainer.tokenizer.save_pretrained(new_model)  # , add_special_tokens=True)


def test_fine_tuned():
    logging.debug("xgen------ Load fine-tuned model ")

    # load it back using the AutoModelForCausalLM class again:
    tokenizer_fine_tuned_xgen = AutoTokenizer.from_pretrained(
        new_model, trust_remote_code=True
    )

    fine_tuned_xgen = AutoModelForCausalLM.from_pretrained("xgen-7b-8k-tuned")

    prompt = "Classify the text into neutral, negative, or positive. Text: Comparable operating profit for the quarter decreased from EUR510m while sales increased from EUR860m , as compared to the third quarter 2007"

    pipe = pipeline(
        task="text-generation",
        model=fine_tuned_xgen,
        tokenizer=tokenizer_fine_tuned_xgen,
        max_length=200,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]["generated_text"])


if DO_TEST:
    test_fine_tuned()

logging.debug("xgen------ Done ")

print(42)
