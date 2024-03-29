import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score

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
import logging
import pandas as pd
from feature_attribution import feature_ablation

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


Path("./data").mkdir(parents=True, exist_ok=True)

# model_name = "xgen-7b-8k-base"
model_name = "xgen-7b-8k-tuned"


base_data_directory = (
    "./data/" + "wavelang_" + model_name + "_{:%Y-%m-%d-%H-%M}".format(now)
)

Path(base_data_directory).mkdir(parents=True, exist_ok=True)
log_file_name = base_data_directory + "/wavelang_{:%Y-%m-%d-%H-%M}.log".format(now)

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


# Attempt to estimate model size in memory - this inference takes a long time
# probabley swapping from gpu to memory
def get_model_size(model):

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size = (param_size + buffer_size) / 1024**2

    return model_size


logging.debug("xgen------ Set device ")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

logging.debug("xgen------ Device = " + device)


logging.debug("xgen------ Load the pre-trained model")

# Load the base or pre-trained model as determined by model_name
if model_name == "xgen-7b-8k-tuned":
    xgen_model = AutoModelForCausalLM.from_pretrained(model_name)
elif model_name == "xgen-7b-8k-base":
    xgen_model = AutoModelForCausalLM.from_pretrained(
        "Salesforce/xgen-7b-8k-base", torch_dtype=torch.bfloat16
    )
else:
    raise "Bad model name : " + model_name + " - must be xgen-7b-8k-tuned or xgen-7b-8k-base"

    # bbcrevisit - we do this in the pipe
    # xgen_model = xgen_model.to(device)

model_size = get_model_size(xgen_model)

logging.info("model size in memory in MB: {:.3f}MB".format(model_size))

logging.debug("xgen------ Load the pre-trained tokenizer")

# Load the pre-trained or base tokenizer as determined by model_name
if model_name == "xgen-7b-8k-tuned":
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
elif model_name == "xgen-7b-8k-base":
    tokenizer = AutoTokenizer.from_pretrained(
        "Salesforce/xgen-7b-8k-base", trust_remote_code=True
    )

logging.debug("xgen------ Load the test set")

samples_per_class = 20
agreement = "TEST"
num_classes = 3


def load_finbert_test_set(samples_per_class):

    # Load the test set
    sentiment_test_df = pd.read_csv(
        "data/finBERT_TRAIN_TEST_VALIDATE/test.csv", sep="\t"
    )
    sentiment_test_df = sentiment_test_df[["text", "label"]]
    num_samples = len(sentiment_test_df)

    num_classes = 3

    min_samples_per_class = 0
    class_distribution = sentiment_test_df["label"].value_counts()
    min_samples_per_class = max(min_samples_per_class, class_distribution.min())

    if min_samples_per_class < samples_per_class:
        samples_per_class = min_samples_per_class
        logging.warn(
            "Reducing samples per class from "
            + str(samples_per_class)
            + " to "
            + str(min_samples_per_class)
        )
    else:
        logging.info("samples_per_class = " + str(samples_per_class))

    # Group the DataFrame by the label column
    grouped_train_df = sentiment_test_df.groupby("label")

    # Sample a specified number of rows from each group
    random_state = 42  # Set a random state for reproducibility
    logging.info("random_state = " + str(random_state))

    sampled_df = grouped_train_df.sample(n=samples_per_class, random_state=random_state)

    # Combine the sampled rows into a new DataFrame
    sentiment_test_df = sampled_df.reset_index(drop=True)
    return sentiment_test_df


sentiment_test_df = load_finbert_test_set(samples_per_class)

logging.debug("xgen------ Create the pipeline")
# Create the pipeline
pipe = pipeline(
    task="text-generation",
    model=xgen_model,
    device=device,
    tokenizer=tokenizer,
    max_length=200,
)


def get_xgen_model_response(prompt):
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]["generated_text"])

    to_parse = result[0]["generated_text"]
    cl_data = to_parse.split("[/INST]")
    ans = cl_data[1]

    if "neutral" in ans:
        ans = "neutral"
    elif "negative" in ans:
        ans = "negative"
    elif "positive" in ans:
        ans = "positive"
    else:
        ans = "NA"

    return ans


system_prompt = " Classify the text into neutral, negative, or positive. Text: "

df = pd.DataFrame(
    columns=[
        "sentence",
        "num_prompt_tokens",
        "xgenft_sentiment",
        "ground_truth",
    ]
)

num_samples = len(sentiment_test_df)
logging.info("num_samples = " + str(num_samples))

for i, data in sentiment_test_df.iterrows():
    if i > num_samples:
        break
    prompt = system_prompt + data["text"]

    num_prompt_tokens = len(prompt.split())

    xgen_model_sentiment = get_xgen_model_response(prompt)

    df.loc[i] = [
        data["text"],
        num_prompt_tokens,
        xgen_model_sentiment,
        data["label"],
    ]

    feature_ablation(
        model=xgen_model.model,
        tokenizer=tokenizer,
        eval_prompt=prompt,
        target=xgen_model_sentiment,
        save_path=base_data_directory,
        sample_id=str(i),
    )


xgen_sentiment_scores = df["xgenft_sentiment"].tolist()

ground_truth_scores = df["ground_truth"].tolist()

kappa_xgenft = cohen_kappa_score(
    xgen_sentiment_scores, ground_truth_scores, weights="linear"
)
xgen_sentiment_classification_crosstab = pd.crosstab(
    df.xgenft_sentiment, df.ground_truth, margins=True
)
xgen_sentiment_classification_crosstab.loc["Grand Total"] = (
    xgen_sentiment_classification_crosstab.sum(numeric_only=True, axis=0)
)
xgen_sentiment_classification_crosstab.loc[:, "Grand Total"] = (
    xgen_sentiment_classification_crosstab.sum(numeric_only=True, axis=1)
)
logging.info(xgen_sentiment_classification_crosstab)

logging.info("xgenft Cohen Kappa Score: " + str("{:.2f}".format(kappa_xgenft)))

logging.info("Saving sentiment classes to csv file")
df.to_csv(base_data_directory + "/xgenft_sentiment_classifications.csv")

logging.info("Saving classification crosstab results to csv file")

xgen_crosstab_filename = (
    base_data_directory
    + "/xgenft_sentiment_crosstab_"
    + "agreement_"
    + agreement
    + "_samples_"
    + str(samples_per_class * num_classes)
    + "_kappa_"
    + str("{:.2f}".format(kappa_xgenft))
    + ".csv"
)

xgen_sentiment_classification_crosstab.to_csv(xgen_crosstab_filename)

logging.info("xgenft Cohen Kappa Score: " + str(kappa_xgenft))

tab = pd.crosstab(df.xgenft_sentiment, df.ground_truth)
xgenft_accuracy = np.diag(tab).sum() / tab.to_numpy().sum()
logging.info("xgenft Accuracy : " + str("{:.2f}".format(xgenft_accuracy)))


def test_xgen(
    text="The contract incorporates a Convergent Charging rating solution for voice and data , which includes Internet , GPRS , SMS , MMS and WAP",
):
    prompt = system_prompt + text

    result = get_xgen_model_response(prompt)

    return result


test_result = test_xgen()
