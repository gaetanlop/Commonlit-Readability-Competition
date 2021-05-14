# Commonlit-Readability

## Interesting notebooks:

* About seeds and cnn heads: https://www.kaggle.com/khoongweihao/tse2020-roberta-cnn-random-seed-distribution
* About LSTM best kernel ever done: https://www.kaggle.com/swarnabha/pytorch-text-classification-torchtext-lstm
* About Freezing embedding layer

## Interesting articles

* https://neptune.ai/blog/text-classification-tips-and-tricks-kaggle-competitions full of tips and tricks
* Look at textstat for FE: https://pypi.org/project/textstat/
* list of good nlp ressources: https://github.com/cedrickchee/awesome-bert-nlp
* LSTM outperforms Bert when small datasets
* add count based features to pretrain models https://arxiv.org/pdf/2010.09078.pdf
* ReadNet: A Hierarchical Transformer Framework for Web Article Readability Analysis https://arxiv.org/pdf/2103.04083v1.pdf
* Get a huge understanging of bert models :https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#sentence-vectors
* 

## Results:

* Random seeds make training unstable. We are unable to get the same result a second time if we are using a different random_state

## Things to try

- Check the impact of the seed 
- Choose one seed and check if using the best number of epochs gives the best results. Rerain on all the dataset.
- Check if stacking last hidden states of concat last hidden states with linear or conv1d head improves performance.
- Choose a good sentence length
- try data augmtentation: from english to french, from french to english
- try to add custom features: https://www.kaggle.com/ravishah1/readability-feature-engineering-non-nn-baseline / https://arxiv.org/pdf/2010.09078.pdf
- Use simple LSTM then concat with mlp hidden layer.

## What I have done

* Create the Dataset for simple MLP
* Bert Embedding + LSTM
* 
