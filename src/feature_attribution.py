import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import sys

from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    LayerIntegratedGradients,
    LLMAttribution,
    LLMGradientAttribution,
    TextTokenInput,
    TextTemplateInput,
    ProductBaselines,
)

import matplotlib.pyplot as plt


def feature_ablation(model, tokenizer, eval_prompt, target, save_path, sample_id):
    
    fa = FeatureAblation(model)

    llm_attr = LLMAttribution(fa, tokenizer)

    inp = TextTokenInput(
        eval_prompt,
        tokenizer,
        skip_tokens=[1],
        # skip the special token for the start of the text <s>
    )
    attr_res = llm_attr.attribute(inp, target="<s> " + target)

    print(
        "attr to the output sequence:",
        attr_res.seq_attr.shape,
        "   ",
        attr_res.seq_attr,
    )  # shape(n_input_token)
    print(
        "attr to the output tokens:",
        attr_res.token_attr.shape,
        "    ",
        attr_res.token_attr,
    )  # shape(n_output_token, n_input_token)

    fig, ax = attr_res.plot_token_attr(show=False)
    fig.savefig("feature_attribution_token.png")
    plt.close(fig)

    attr_res = llm_attr.attribute(inp, target=target)

    fig, ax = attr_res.plot_token_attr(show=False)
    fig.savefig("feature_attribution_keyword.png")
    plt.close(fig)


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


if __name__ == "__main__":
        
    def test_llama_2_13b_feature_attribution():

        model_name = "meta-llama/Llama-2-13b-chat-hf"

        bnb_config = create_bnb_config()

        model, tokenizer = load_Llama_model(model_name, bnb_config)

        eval_prompt = (
            "Dave lives in Palm Coast, FL and is a lawyer. His personal interests include"
        )

        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(model_input["input_ids"], max_new_tokens=30)[0]
            response = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(response)

        # #the `FeatureAblation` attribution result of our LLM.
        # The return contains the attribution tensors to both the entire generated target seqeuence
        # and each generated token, which tell us how each input token impact the output and each token within it.
        fa = FeatureAblation(model)

        llm_attr = LLMAttribution(fa, tokenizer)

        inp = TextTokenInput(
            eval_prompt,
            tokenizer,
            skip_tokens=[1],  # skip the special token for the start of the text <s>
        )

        target0 = "playing guitar, hiking, and spending time with his family."
        target = "golf, travel, and spending time with his family."

        # target = "Dave is a member of the Flagler County Bar Association and the Florida Bar"

        attr_res = llm_attr.attribute(inp, target=target)

        print(
            "attr to the output sequence:",
            attr_res.seq_attr.shape,
            "   ",
            attr_res.seq_attr,
        )  # shape(n_input_token)
        print(
            "attr to the output tokens:",
            attr_res.token_attr.shape,
            "    ",
            attr_res.token_attr,
        )  # shape(n_output_token, n_input_token)

        import matplotlib.pyplot as plt

        fig, ax = attr_res.plot_token_attr(show=False)
        fig.savefig("feature_attribution_token.png")
        plt.close(fig)

        # Keyword attribution
        inp = TextTemplateInput(
            template="{} lives in {}, {} and is a {}. {} personal interests include",
            values=["Dave", "Palm Coast", "FL", "lawyer", "His"],
        )

        target = "golf, travel, family"

        attr_res = llm_attr.attribute(inp, target=target)

        fig, ax = attr_res.plot_token_attr(show=False)
        fig.savefig("feature_attribution_keyword.png")
        plt.close(fig)


    test_llama_2_13b_feature_attribution()

    
