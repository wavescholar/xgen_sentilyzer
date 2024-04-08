"""
Comparing Sentiment Analysis with Gemini FinBert and OpenAI models

Uses the FinBert training set : https://huggingface.co/datasets/financial_phrasebank
See the original paper "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts"
Here : https://arxiv.org/abs/1307.5336

We use weighted Cohen's Kappa to compare the results.
See here for more info:

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
https://www.datanovia.com/en/lessons/weighted-kappa-in-r-for-two-ordinal-variables/

"""

import os
import sys
import numpy as np
import json
from random import randint
from time import sleep
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

from huggingface_utils import load_huggingface_data, prep_huggingface_data
from finbert_utils import get_finbert_response
from openai_utils import get_openai_response
from google_gemini_utils import get_gemini_response, get_gemini_model


from finbert_utils import load_finbert_test_set

from llama7b_utils import get_llama_response

now = datetime.now()

Path("./data").mkdir(parents=True, exist_ok=True)
base_data_directory = "./data/" + "wavelang_experiment_{:%Y-%m-%d-%H-%M}".format(now)

Path(base_data_directory).mkdir(parents=True, exist_ok=True)

from sentylizer_utils import logging_config

logger = logging_config(base_data_directory=base_data_directory, now=now)

# Mapping from dataset class label to LLM sentiment responses - used by Gemini,OpenAI and the finbert inference
label_map = {"0": "Negative", "1": "Neutral", "2": "Positive"}


def run_kappa_calculation(
    samples_per_class,
    agreement,
    k_shot,
    openai_model,
    gemini_model,
    api_sleep_min,
    api_sleep_max,
):
    """[summary] run_kappa_calculation

    Args:
        samples_per_class ([type]): [description]
        agreement ([type]): [description]
    """
    gemini_chat = get_gemini_model(gemini_model)

    if agreement != "TEST":
        # Load the Huggingface dataset
        eval_df = load_huggingface_data(agreement=agreement)

        eval_df, k_shot_df = prep_huggingface_data(
            eval_df,
            samples_per_class=samples_per_class,
            k_shot=k_shot,
            base_data_directory=base_data_directory,
            now=now,
        )
    else:
        logging.info("Loading finbert  test data")
        num_classes = 3
        eval_df = load_finbert_test_set(samples_per_class + num_classes * k_shot)

        if k_shot > 0:
            k_shot_samples = k_shot * num_classes
            # Peel off the number needed for k_shot prompt
            eval_df, k_shot_df = train_test_split(
                eval_df[["sentence", "label"]],
                stratify=eval_df["label"],
                train_size=num_samples,
                test_size=k_shot_samples,
            )
            eval_df = eval_df.reset_index(drop=True)
            k_shot_df = k_shot_df.reset_index(drop=True)

    logging.info("Head data")
    for i, data in eval_df.iterrows():
        if i > 10:
            break
        logging.info(str(i) + "---" + data["sentence"] + "---" + str(data["label"]))

    system_prompt = "Classify the sentiment of the following news text into neutral, negative, or positive. "

    # Prepare the k_shot prompt if needed
    k_shot_prompt = system_prompt
    if k_shot > 0:
        for i, data in k_shot_df.iterrows():
            k_shot_prompt += (
                str(i + 1)
                + ". Text :"
                + data["sentence"]
                + os.linesep
                + "Sentiment: "
                + label_map[data["label"]]
                + os.linesep
            )

    df = pd.DataFrame(
        columns=[
            "sentence",
            "num_prompt_tokens",
            "gemini_sentiment",
            "openai_sentiment",
            "finbert_sentiment",
            "llama_sentiment",
            "ground_truth",
        ]
    )

    num_samples = len(eval_df)

    for i, data in eval_df.iterrows():
        if i > num_samples:
            break

        prompt = (
            k_shot_prompt
            + os.linesep
            + " Text: "
            + data["sentence"]
            + os.linesep
            + " Sentiment: "
        )

        num_prompt_tokens = len(prompt.split())

        openai_response = get_openai_response(
            k_shot_prompt,
            os.linesep + " Text: " + data["sentence"] + os.linesep + " Sentiment: ",
            openai_model,
        )

        logging.info(
            "OpenAI--"
            + prompt
            + "   "
            + openai_response
            + " ground truth = "
            + label_map[data["label"]]
        )

        gemini_response = get_gemini_response(gemini_chat, prompt)

        logging.info(
            "Gemini--"
            + prompt
            + "   "
            + gemini_response
            + " ground truth = "
            + label_map[data["label"]]
        )

        # We don't use k_shot here
        finbert_response = get_finbert_response(data["sentence"])

        logging.info(
            "FinBERT--"
            + data["sentence"]
            + "   "
            + finbert_response
            + " ground truth = "
            + label_map[data["label"]]
        )

        # Llama gives poor results when k=0
        llama_response = get_llama_response(prompt)

        logging.info(
            "Llama--"
            + k_shot_prompt
            + "   "
            + llama_response
            + " ground truth = "
            + label_map[data["label"]]
        )

        df.loc[i] = [
            data["sentence"],
            num_prompt_tokens,
            gemini_response,
            openai_response,
            finbert_response,
            llama_response,
            label_map[data["label"]],
        ]

        sleep(randint(api_sleep_min, api_sleep_max))

    # -------------------
    gemini_sentiment_scores = df["gemini_sentiment"].tolist()

    ground_truth_scores = df["ground_truth"].tolist()

    kappa_gemini = cohen_kappa_score(
        gemini_sentiment_scores, ground_truth_scores, weights="linear"
    )

    logging.info("Gemini Cohen Kappa Score: " + str("{:.2f}".format(kappa_gemini)))

    # -----------------------
    openai_sentiment_scores = df["openai_sentiment"].tolist()

    kappa_openai = cohen_kappa_score(
        openai_sentiment_scores, ground_truth_scores, weights="linear"
    )

    logging.info("OpenAI Cohen Kappa Score: " + str(kappa_openai))

    # ------------------------
    finbert_sentiment_scores = df["finbert_sentiment"].tolist()

    kappa_finbert = cohen_kappa_score(
        finbert_sentiment_scores, ground_truth_scores, weights="linear"
    )

    logging.info("Finbert Cohen Kappa Score: " + str(kappa_finbert))

    # ------------------------
    llama_sentiment_scores = df["llama_sentiment"].tolist()

    kappa_llama = cohen_kappa_score(
        llama_sentiment_scores, ground_truth_scores, weights="linear"
    )

    logging.info("Llama Cohen Kappa Score: " + str(kappa_llama))

    # -----------------
    gemini_sentiment_classification_crosstab = pd.crosstab(
        df.gemini_sentiment, df.ground_truth, margins=True
    )
    gemini_sentiment_classification_crosstab.loc["Grand Total"] = (
        gemini_sentiment_classification_crosstab.sum(numeric_only=True, axis=0)
    )
    gemini_sentiment_classification_crosstab.loc[:, "Grand Total"] = (
        gemini_sentiment_classification_crosstab.sum(numeric_only=True, axis=1)
    )
    logging.info(gemini_sentiment_classification_crosstab)

    # -----------------
    openai_sentiment_classification_crosstab = pd.crosstab(
        df.openai_sentiment, df.ground_truth, margins=True
    )
    openai_sentiment_classification_crosstab.loc["Grand Total"] = (
        openai_sentiment_classification_crosstab.sum(numeric_only=True, axis=0)
    )
    openai_sentiment_classification_crosstab.loc[:, "Grand Total"] = (
        openai_sentiment_classification_crosstab.sum(numeric_only=True, axis=1)
    )
    logging.info(openai_sentiment_classification_crosstab)

    # -----------------
    finbert_sentiment_classification_crosstab = pd.crosstab(
        df.finbert_sentiment, df.ground_truth, margins=True
    )
    finbert_sentiment_classification_crosstab.loc["Grand Total"] = (
        finbert_sentiment_classification_crosstab.sum(numeric_only=True, axis=0)
    )
    finbert_sentiment_classification_crosstab.loc[:, "Grand Total"] = (
        finbert_sentiment_classification_crosstab.sum(numeric_only=True, axis=1)
    )
    logging.info(finbert_sentiment_classification_crosstab)

    # -----------------
    llama_sentiment_classification_crosstab = pd.crosstab(
        df.llama_sentiment, df.ground_truth, margins=True
    )
    llama_sentiment_classification_crosstab.loc["Grand Total"] = (
        llama_sentiment_classification_crosstab.sum(numeric_only=True, axis=0)
    )
    llama_sentiment_classification_crosstab.loc[:, "Grand Total"] = (
        llama_sentiment_classification_crosstab.sum(numeric_only=True, axis=1)
    )
    logging.info(llama_sentiment_classification_crosstab)

    logging.info("Saving sentiment classes to csv file")
    df.to_csv(base_data_directory + "/sentiment_classifications.csv")

    logging.info("Saving classification crosstab results to csv file")

    gemini_crosstab_filename = (
        base_data_directory
        + "/gemini_sentiment_crosstab_"
        + "agreement_"
        + agreement
        + "_samples_"
        + str(samples_per_class * num_classes)
        + "_kappa_"
        + str("{:.2f}".format(kappa_gemini))
        + "_k-shot_"
        + str(k_shot)
        + ".csv"
    )

    gemini_sentiment_classification_crosstab.to_csv(gemini_crosstab_filename)

    openai_crosstab_filename = (
        base_data_directory
        + "./openai_sentiment_crosstab_"
        + "agreement_"
        + agreement
        + "_samples_"
        + str(samples_per_class * num_classes)
        + "_kappa_"
        + str("{:.2f}".format(kappa_openai))
        + "_k-shot_"
        + str(k_shot)
        + ".csv"
    )

    openai_sentiment_classification_crosstab.to_csv(openai_crosstab_filename)

    finbert_crosstab_filename = (
        base_data_directory
        + "./finbert_sentiment_crosstab_"
        + "agreement_"
        + agreement
        + "_samples_"
        + str(samples_per_class * num_classes)
        + "_kappa_"
        + str("{:.2f}".format(kappa_finbert))
        + "_k-shot_"
        + str(k_shot)
        + ".csv"
    )

    finbert_sentiment_classification_crosstab.to_csv(finbert_crosstab_filename)

    llama_crosstab_filename = (
        base_data_directory
        + "./llama_sentiment_crosstab_"
        + "agreement_"
        + agreement
        + "_samples_"
        + str(samples_per_class * num_classes)
        + "_kappa_"
        + str("{:.2f}".format(kappa_llama))
        + "_k-shot_"
        + str(k_shot)
        + ".csv"
    )

    llama_sentiment_classification_crosstab.to_csv(llama_crosstab_filename)

    # Accuracy Calculation
    tab = pd.crosstab(df.finbert_sentiment, df.ground_truth)
    finbert_accuracy = np.diag(tab).sum() / tab.to_numpy().sum()
    logging.info("Finbert Accuracy : " + str("{:.2f}".format(finbert_accuracy)))

    tab = pd.crosstab(df.openai_sentiment, df.ground_truth)
    openai_accuracy = np.diag(tab).sum() / tab.to_numpy().sum()
    logging.info("OpenAI Accuracy : " + str("{:.2f}".format(openai_accuracy)))

    tab = pd.crosstab(df.gemini_sentiment, df.ground_truth)
    gemini_accuracy = np.diag(tab).sum() / tab.to_numpy().sum()
    logging.info("Gemini Accuracy : " + str("{:.2f}".format(gemini_accuracy)))

    tab = pd.crosstab(df.llama_sentiment, df.ground_truth)
    llama_accuracy = np.diag(tab).sum() / tab.to_numpy().sum()
    logging.info("Llama Accuracy : " + str("{:.2f}".format(llama_accuracy)))

    logging.info("Done.")


if __name__ == "__main__":

    logging.info("wavelang driver : args ")

    # Load the configuration file
    with open("config.json", "r") as f:
        config = json.load(f)

    logger.info("JSON Parameters : ")
    logger.info(json.dumps(config, indent=4))

    if config["system"]["log_level"] == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    elif config["system"]["log_level"] == "INFO":
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)

    # Getting configuration data
    api_sleep_min = config["system"]["api_sleep_min"]
    api_sleep_max = config["system"]["api_sleep_max"]
    logging.info("api_sleep_min : " + str(api_sleep_min))
    logging.info("api_sleep_max : " + str(api_sleep_max))

    k_shot = config["model_params"]["k_shot"]
    logging.info("k shot k : " + str(k_shot))

    openai_model = config["model_params"]["openai"]["openai_model"]
    logging.info("openai_model : " + str(openai_model))

    gemini_model = config["model_params"]["gogle_gemini"]["gemini_model"]
    logging.info("gemini_model : " + str(gemini_model))

    # Find max value for samples_per_class
    agreements = [
        "sentences_50agree",
        "sentences_66agree",
        "sentences_75agree",
        "sentences_allagree",
    ]
    min_samples_per_class = 0
    for agreement in agreements:
        eval_df = load_huggingface_data(agreement=agreement)
        class_distribution = eval_df["label"].value_counts()
        min_samples_per_class = max(min_samples_per_class, class_distribution.min())

    samples_per_class = config["system"]["samples_per_class"]
    agreement_set = config["system"]["agreement"]

    if config["system"]["samples_per_class"] > min_samples_per_class:
        logging.warn(
            "Overriding samples_per_class to min value : " + str(min_samples_per_class)
        )
        samples_per_class = min_samples_per_class

    logging.info("samples_per_class : " + str(samples_per_class))

    if agreement_set == "RUN_ALL":
        for agreement in agreements:
            logging.info("Running agreement : " + agreement)
            run_kappa_calculation(
                samples_per_class,
                agreement,
                k_shot,
                openai_model,
                gemini_model,
                api_sleep_min,
                api_sleep_max,
            )
    elif agreement_set == "TEST":
        run_kappa_calculation(
            samples_per_class,
            "TEST",
            k_shot,
            openai_model,
            gemini_model,
            api_sleep_min,
            api_sleep_max,
        )
    else:
        run_kappa_calculation(
            samples_per_class,
            agreement,
            k_shot,
            openai_model,
            gemini_model,
            api_sleep_min,
            api_sleep_max,
        )

    logging.info("Done.")
