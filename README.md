# wavelang

This repo is for testing out some large language models on sentiment data.

We test OpenAI, Google Gemini, a fine tuned BERT model, and Saleforce's open source xgen model. We're comparing the ability to classify financial sentiment data into three ordinal classes negative, netrual, and positive. The data is labeled by independent experts and is partitioned into four levels of agreement:
sentences_50agree
sentences_66agree
sentences_75agree
sentences_allagree
The original data is loced here:
https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10

There are three main scripts

1. xgen_finetune.py
2. run_sentiment_experiment.py
3. xgen_inference.py

xgen_inference is run on it's own while the rest of the models are run in the experiment script.
The fine tuned xgen model does not fit in the developers GPU (RTX 3080 ti with 12GB vRAM).

We started work on applying feature attribution to understand how the models are making their decisions. For now it only works on Llama

For now there are two python environments to manage. There's an issue with the transformer library the requires pinning it for the fine tuning.

The get_finbert_data_split in finber_utils.py gives us the train test validate split that was used to train finbert. Use the 50% argree data in 251231364_FinancialPhraseBank-v10 as the source. See the finbert github repo for more details.

When comparing xgen fine tuned results we use the test set from the 50% agreemnt data. There's an option for using the data stored on Huggingface which has all the levels of agreement. It's useful to see how performance varies according to the level of agreement.

### Dockerbuild

```
docker build -t wavelang -f ./Dockerfile .
docker tag wavelang wavescholar/wavelang:latest
docker push -a wavescholar/wavelang
```
