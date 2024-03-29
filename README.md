# XGen Financial Sentiment 

This repo is for fine-tuning a Saleforce xgen model financial sentiment data.  We then test it out against various large language models.

We test OpenAI, Google Gemini, a fine-tuned BERT model, and Saleforce's open-source xgen model. We're comparing the ability to classify financial sentiment data into three ordinal classes: negative, neutral, and positive. The data is labeled by independent experts and is partitioned into four levels of agreement:
- sentences_50agree
- sentences_66agree
- sentences_75agree
- sentences_allagree

The original data is located here:
https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10

There are three main scripts

1. xgen_finetune.py
2. run_sentiment_experiment.py
3. xgen_inference.py

xgen_inference is run on its own while the rest of the models are run in the experiment script. The reason for this is the fine-tuned Xgen model does not fit in the developer's GPU (RTX 3080 ti with 12GB vRAM), and inference takes a long time. 

We started work on applying feature attribution to understand how the models are making their decisions. For now, it only works on Llama

For now, there are two Python environments to manage. The transformer library has an issue that requires pinning it for fine-tuning.

The get_finbert_data_split in finber_utils.py gives us the train test validate split used to train Finbert. Use the 50% agree data in 251231364_FinancialPhraseBank-v10 as the source. See the Finbert GitHub repo [ https://github.com/ProsusAI/finBERT ] for more details.

When comparing xgen fine-tuned results, we use the test set from the 50% agreement data. There's an option for using the data stored on Huggingface, which has all the levels of agreement. It's useful to see how performance varies according to the level of agreement.



