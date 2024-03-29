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

import argparse
import sys
import numpy as np
import json

from huggingface_utils import load_huggingface_data
from finbert_utils import get_finbert_response
from openai_utils import get_openai_response
from google_gemini_utils import get_gemini_response, get_gemini_model


import pandas as pd
from sklearn.metrics import cohen_kappa_score

from random import randint
from time import sleep

import logging

# get date and time
from datetime import datetime

from pathlib import Path


# from xgen_utils import load_finbert_test_set
def load_finbert_test_set(samples_per_class):

    # Load the test set
    sentiment_test_df = pd.read_csv(
        "data/finBERT_TRAIN_TEST_VALIDATE/test.csv", sep="\t"
    )
    sentiment_test_df = sentiment_test_df[["text", "label"]]

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


now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

Path("./data").mkdir(parents=True, exist_ok=True)
base_data_directory = "./data/" + "wavelang_experiment_{:%Y-%m-%d-%H-%M}".format(now)
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

logging.debug("wavelang log file ")

# Mapping from dataset class label to LLM sentiment responses - used by Gemini,OpenAI and the finbert inference
label_map = {"0": "Negative", "1": "Neutral", "2": "Positive"}

logging.info("date and time =" + dt_string)
logging.info("---------" + dt_string + "----------")


def run_kappa_calculation(
    samples_per_class,
    agreement,
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

    num_classes = 3

    if agreement != "TEST":
        # Load the Huggingface dataset
        eval_df = load_huggingface_data(agreement=agreement)

        class_distribution = eval_df["label"].value_counts()

        num_classes = len(class_distribution)
        logging.info("num_classes = " + str(num_classes))

        logging.info("Class distribution")
        logging.info(class_distribution)

        logging.info("samples_per_class = " + str(samples_per_class))

        num_samples = num_classes * samples_per_class

        # Group the DataFrame by the label column
        grouped_train_df = eval_df.groupby("label")

        # Sample a specified number of rows from each group
        random_state = 42  # Set a random state for reproducibility
        logging.info("random_state = " + str(random_state))

        sampled_df = grouped_train_df.sample(
            n=samples_per_class, random_state=random_state
        )

        # Combine the sampled rows into a new DataFrame
        eval_df = sampled_df.reset_index(drop=True)
        eval_df.to_csv(
            base_data_directory
            + "/wavelang_eval_data_{:%Y-%m-%d-%H-%M}.csv".format(now)
        )
    else:
        logging.info("Loading finbert  test data")
        eval_df = load_finbert_test_set(samples_per_class)
        # Now rename a column to be consistent
        eval_df.rename(columns={"text": "sentence"}, inplace=True)

        # The training data class labels as given by the original data set
        # do not match the Huggin face dataset -  we need to map
        label_map_training_data = {"negative": "0", "neutral": "1", "positive": "2"}
        for i, data in eval_df.iterrows():
            eval_df.at[i, "label"] = label_map_training_data[data["label"]]

    logging.info("Head Training data")
    for i, data in eval_df.iterrows():
        if i > 10:
            break
        logging.info(str(i) + "---" + data["sentence"] + "---" + str(data["label"]))

    zero_shot_prompt_system = (
        "Classify the text into neutral, negative, or positive. Text: "
    )

    df = pd.DataFrame(
        columns=[
            "sentence",
            "num_prompt_tokens",
            "gemini_sentiment",
            "openai_sentiment",
            "finbert_sentiment",
            "ground_truth",
        ]
    )

    num_samples = len(eval_df)

    for i, data in eval_df.iterrows():
        if i > num_samples:
            break

        prompt = zero_shot_prompt_system + data["sentence"]

        num_prompt_tokens = len(prompt.split())

        openai_response = get_openai_response(
            zero_shot_prompt_system, data["sentence"], openai_model
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

        finbert_response = get_finbert_response(data["sentence"])

        logging.info(
            "FinBERT--"
            + prompt
            + "   "
            + finbert_response
            + " ground truth = "
            + label_map[data["label"]]
        )

        df.loc[i] = [
            data["sentence"],
            num_prompt_tokens,
            gemini_response,
            openai_response,
            finbert_response,
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
        + ".csv"
    )

    finbert_sentiment_classification_crosstab.to_csv(finbert_crosstab_filename)

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

    logging.info("Done.")


if __name__ == "__main__":

    logging.info("wavelang driver : args ")
    for arg in sys.argv:
        logging.info(arg)

    class ParseKwargs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split("=")
                getattr(namespace, self.dest)[key] = value

    # Pass in args like this --verbose --kwargs api_key='kv' things2='val' ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="enable verbose output")
    parser.add_argument(
        "--samples_per_class", help="number of examples from each class", type=int
    )
    parser.add_argument(
        "--agreement",
        help="Percentage of agreement of annotators use RUN_ALL to run them all : sentences_50agree Number| sentences_66agree |sentences_75agree |sentences_allagree",
    )

    args = parser.parse_args()
    logger.info("Args parsed : " + str(args))

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Load the configuration file
    with open("config.json", "r") as f:
        config = json.load(f)

    # Getting configuration data
    api_sleep_min = config["system"]["api_sleep_min"]
    api_sleep_max = config["system"]["api_sleep_max"]
    logging.info("api_sleep_min : " + str(api_sleep_min))
    logging.info("api_sleep_max : " + str(api_sleep_max))

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

    if args.samples_per_class > min_samples_per_class:
        logging.warn(
            "Overriding samples_per_class to min value : " + str(min_samples_per_class)
        )
        args.samples_per_class = min_samples_per_class

    logging.info("samples_per_class : " + str(args.samples_per_class))

    if args.agreement == "RUN_ALL":
        for agreement in agreements:
            logging.info("Running agreement : " + agreement)
            run_kappa_calculation(
                args.samples_per_class,
                agreement,
                openai_model,
                gemini_model,
                api_sleep_min,
                api_sleep_max,
            )
    elif args.agreement == "TEST":
        run_kappa_calculation(
            args.samples_per_class,
            "TEST",
            openai_model,
            gemini_model,
            api_sleep_min,
            api_sleep_max,
        )
    else:
        run_kappa_calculation(
            args.samples_per_class,
            agreement,
            openai_model,
            gemini_model,
            api_sleep_min,
            api_sleep_max,
        )

    logging.info("Done.")
