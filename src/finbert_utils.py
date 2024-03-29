from transformers import AutoTokenizer, AutoModelForSequenceClassification

finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
import pandas as pd

import logging


# Set the number of samples per class for the training set and the number of classes in the dataset

samples_per_class = 20
agreement = "TEST"
num_classes = 3


def load_finbert_train_set(samples_per_class):

    # Load the test set
    sentiment_test_df = pd.read_csv(
        "data/finBERT_TRAIN_TEST_VALIDATE/train.csv", sep="\t"
    )
    sentiment_test_df = sentiment_test_df[["text", "label"]]
    num_samples = len(sentiment_test_df)  # Get the number of samples in the test set

    # Calculate the minimum number of samples per class
    # This is to ensure that we have a balanced dataset

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


k = 3
samples_per_class = 1
train_df = load_finbert_train_set(1)

zero_shot_prompt_system = (
    "Classify the text into neutral, negative, or positive. Text: "
)


def get_finbert_response(sentence):
    """Predict the sentiment of a sentence using the FinBERT model.

    Args:
          sentence ([type]): The sentence to predict the sentiment of.

    Raises:
        ValueError: [description]

    Returns:
        The predicted sentiment of the sentence.

        int: The predicted sentiment of the sentence. 0 for negative, 1 for neutral, and 2 for positive.
    """
    input_ids = finbert_tokenizer.encode(sentence, return_tensors="pt")
    outputs = finbert_model(input_ids)
    logits = outputs.logits
    probs = logits.softmax(dim=-1)

    probs = logits.softmax(dim=-1)
    logging.info("finBert probs " + str(probs))

    class_val = probs.argmax().item()

    # These are not consistent with the training data. The label mapping is different for the model
    # - something to investigate
    if class_val == 0:
        class_label = "Positive"
    elif class_val == 1:
        class_label = "Negative"
    elif class_val == 2:
        class_label = "Neutral"
    else:
        raise ValueError("Invalid class value: {}".format(class_val))

    logging.info("finBert class_val " + str(class_val))
    logging.info("finBert class_label " + str(class_label))

    return class_label


# This is the split code from the finbert repo. The code operates on the dataset
# which is available here : https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def get_finbert_data_split():

    if not os.path.exists("data/sentiment_data"):
        os.makedirs("data/sentiment_data")
    os.path.exists("data/sentiment")
    parser = argparse.ArgumentParser(description="Sentiment analyzer")
    parser.add_argument("--data_path", type=str, help="Path to the text file.")

    args = parser.parse_args()
    data_path = args.data_path
    data_path = "data/sentiment/Sentences_50Agree.txt"
    data = pd.read_csv(data_path, sep=".@", names=["text", "label"])

    train, test = train_test_split(data, test_size=0.2, random_state=0)
    train, valid = train_test_split(train, test_size=0.1, random_state=0)

    train.to_csv("data/sentiment_data/train.csv", sep="\t")
    test.to_csv("data/sentiment_data/test.csv", sep="\t")
    valid.to_csv("data/sentiment_data/validation.csv", sep="\t")
