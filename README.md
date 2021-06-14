# Commonlit-Readability


## What I have done

* 07/06/2021-08/06/2021: create a simple training script for BERT. 
* 08/06/2021: Add Multi Sample Dropout based on this paper https://arxiv.org/pdf/1905.09788.pdf
* 12/06/2021: Read the paper https://arxiv.org/pdf/1905.05583.pdf + creation of custom Pytorch Trainer and Evaluator for Roberta. Pipeline finished.
* 13/06/2021: Bert pretraining on our corpus. Still did not succeed to do it. Will finish it tomorrow.
* 14/06/2021: Create a Script to validate the model mutlitple times in one epoch to fight against instability 
* 15/06/2021: Goal learn differential learning rate and apply it.

## Things to try

- Multi Sample Dropout not working well
- Mean Max head: not working well
- try convolutional heads
- try attention head
- try pseudo labeling as the dataset is very small
- More regularization techniques like batchnorm, layer norm ...
- pretrain bert on our corpus
- Validate multiple times in one epoch
- Use Distillbert (smaller model)
- character level bert / sentence level bert
- Check the impact of the seed https://www.kaggle.com/khoongweihao/tse2020-roberta-cnn-random-seed-distribution
- try to add custom features: https://www.kaggle.com/ravishah1/readability-feature-engineering-non-nn-baseline / https://arxiv.org/pdf/2010.09078.pdf
- Use simple LSTM then concat with mlp hidden layer (no one do it but good results in two papers about it. I will try it after tuning the Bert Model).
- try data augmtentation: from english to french, from french to english

## To Study

- Global average pooling

## Interesting notebooks to look at:

* About seeds and cnn heads: https://www.kaggle.com/khoongweihao/tse2020-roberta-cnn-random-seed-distribution
* About LSTM best kernel ever done: https://www.kaggle.com/swarnabha/pytorch-text-classification-torchtext-lstm
* About Freezing embedding layer
* Add features to Bert: https://github.com/Anushka-Prakash/RumourEval-2019-Stance-Detection/

## Interesting articles for the competition

* https://neptune.ai/blog/text-classification-tips-and-tricks-kaggle-competitions full of tips and tricks
* Look at textstat for FE: https://pypi.org/project/textstat/
* list of good nlp ressources: https://github.com/cedrickchee/awesome-bert-nlp
* LSTM outperforms Bert when small datasets
* add count based features to pretrain models https://arxiv.org/pdf/2010.09078.pdf
* ReadNet: A Hierarchical Transformer Framework for Web Article Readability Analysis https://arxiv.org/pdf/2103.04083v1.pdf
* Get a huge understanging of bert models :https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#sentence-vectors


## Best resource to learn transformers
https://jalammar.github.io/illustrated-transformer/
