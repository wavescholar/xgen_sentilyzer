import pandas as pd

from huggingface_hub import hf_hub_download
from datasets import load_dataset


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
