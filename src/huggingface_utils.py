from datetime import datetime
import logging
import pandas as pd

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def prep_huggingface_data(
    eval_df, samples_per_class=10, k_shot=1, base_data_directory="data", now=None
):

    num_classes = 3
    num_samples = num_classes * samples_per_class
    logging.info("num_samples = " + str(num_samples))

    k_shot_samples = k_shot * num_classes
    logging.info("k_shot_samples = " + str(k_shot_samples))

    class_distribution = eval_df["label"].value_counts()

    num_classes = len(class_distribution)
    logging.info("num_classes = " + str(num_classes))

    logging.info("Class distribution")
    logging.info(class_distribution)

    logging.info("samples_per_class = " + str(samples_per_class))

    # Group the DataFrame by the label column
    grouped_train_df = eval_df.groupby("label")

    # Sample a specified number of rows from each group
    random_state = 42  # Set a random state for reproducibility
    logging.info("random_state = " + str(random_state))

    # Sample the number needed for eval plus the prompt
    sample_df = grouped_train_df.sample(
        n=samples_per_class + k_shot, random_state=random_state
    )
    eval_df = sample_df.reset_index(drop=True)

    k_shot_df = None
    if k_shot > 0:
        # Peel off the number needed for k_shot prompt
        eval_df, k_shot_df = train_test_split(
            eval_df[["sentence", "label"]],
            stratify=eval_df["label"],
            train_size=num_samples,
            test_size=k_shot_samples,
        )
        eval_df = eval_df.reset_index(drop=True)
        k_shot_df = k_shot_df.reset_index(drop=True)
        k_shot_df.to_csv(
            base_data_directory
            + "/wavelang_k_shot_data_{:%Y-%m-%d-%H-%M}.csv".format(now)
        )

    eval_df.to_csv(
        base_data_directory + "/wavelang_eval_data_{:%Y-%m-%d-%H-%M}.csv".format(now)
    )

    return eval_df, k_shot_df


def load_huggingface_data(agreement="sentences_75agree"):
    """loads the Hugginface dataset and converts it from Arrow to Pandas

    sentences_50agree Number of instances with >=50% annotator agreement: 4846
    sentences_66agree: Number of instances with >=66% annotator agreement: 4217
    sentences_75agree: Number of instances with >=75% annotator agreement: 3453
    sentences_allagree: Number of instances with 100% annotator agreement: 2264
    Args:
        agreement (str, optional): [description]. Defaults to "sentences_75agree".
    """
    dataset = load_dataset("financial_phrasebank", agreement)
    train = dataset["train"]

    train_df = pd.DataFrame(train.to_pandas())
    train_df["label"] = train_df["label"].astype(str)
    return train_df
