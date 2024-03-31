# XGen Financial Sentiment 

This repo is for fine-tuning a Saleforce xgen model financial sentiment data and evaluating it against various large language models.

We test OpenAI, Google Gemini, a fine-tuned BERT model, and Saleforce's open-source xgen model. We're comparing the ability to classify financial sentiment data into three ordinal classes: negative, neutral, and positive. A group of 16 individuals, all possessing relevant expertise in financial markets, carried out annotations on a curated set of phrases. This group comprised three researchers and thirteen master's students from Aalto University School of Business, specializing mainly in finance, accounting, and economics. Due to the significant overlap in annotations, with each sentence receiving between 5 to 8 annotations, multiple approaches exist for establishing a gold standard through majority consensus. There are four distinct reference datasets, each derived from the degree of majority concurrence.
- sentences_50agree
- sentences_66agree
- sentences_75agree
- sentences_allagree

*Malo, P., Sinha, A., Takala, P., Korhonen, P. and Wallenius, J. (2013): “Good debt or bad debt: Detecting semantic orientations in economic texts.” Journal of the American Society for Information Science and Technology.*

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

Cohen's Weighted Kappa was used as a metric for comparing the ordinal classification capabilities of the models. This is better than simply reporting accuracy, which we also record. Cohen's weighted Kappa takes into account that it would be worse to classify something negative as positive than to classify something negative as neutral. 
See here for more;

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

https://www.datanovia.com/en/lessons/weighted-kappa-in-r-for-two-ordinal-variables/

# Results

This is a summary of the results. The results folder contains selected experiments with crosstabs and logs. The initial sample size reported here is a very minimal 30 due to lack of vRAM for GPU. We note OpenAI had trouble distinguishing neutral text in many of the experiments. We'll update to GPT-4 in the future. 

| Model                 | Kappa  | Accuracy  |   
|-----------------------|--------|-----------|
| Finbert               |  0.77  |  0.83     |   
| Google Gemini         |  0.72  |  0.73     |   
| Open AI GPT-3.5Turbo  |  0.43  |  0.57     |   
| Xgen_fine Tuned       |  0.92  |  0.93     |   
