import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import pandas as pd

finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")


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

    # Now rename a column to be consistent
    sentiment_test_df.rename(columns={"text": "sentence"}, inplace=True)

    # The training data class labels as given by the original data set
    # do not match the Hugging face dataset -  we need to map
    label_map_training_data = {"negative": "0", "neutral": "1", "positive": "2"}
    for i, data in sentiment_test_df.iterrows():
        sentiment_test_df.at[i, "label"] = label_map_training_data[data["label"]]

    return sentiment_test_df
