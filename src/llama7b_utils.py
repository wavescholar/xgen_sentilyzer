import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_Llama_model(model_name, bnb_config):
    n_gpus = torch.cuda.device_count()
    max_memory = "10000MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
        max_memory={i: max_memory for i in range(n_gpus)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


model_name = "meta-llama/Llama-2-13b-chat-hf"

bnb_config = create_bnb_config()

model, tokenizer = load_Llama_model(model_name, bnb_config)


def get_llama_response(eval_prompt):

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(model_input["input_ids"], max_new_tokens=30)[0]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        sep = response.split("Sentiment:")
        response = sep[len(sep) - 1].strip()
        if response.find("Positive"):
            response = "Positive"
        if response.find("Negative"):
            response = "Negative"
        if response.find("Neutral"):
            response = "Neutral"
    return response
