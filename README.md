# Commonlit-Readability


## What I have done

* 07/06/2021-08/06/2021: create a simple training script for BERT. 
* 08/06/2021: Add Multi Sample Dropout based on this paper https://arxiv.org/pdf/1905.09788.pdf
* 12/06/2021: Read the paper https://arxiv.org/pdf/1905.05583.pdf + creation of custom Pytorch Trainer and Evaluator for Roberta. Pipeline finished.
* 13/06/2021: Bert pretraining on our corpus. Still did not succeed to do it. Will finish it tomorrow.
* 14/06/2021: Create a Script to validate the model mutlitple times in one epoch to fight against instability : 0.5299962748524167
* 15/06/2021: Implemented differential learning rate based on https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-fit?scriptVersionId=64693127 + freezing/unfreezing.Learned about Pretrain on task dataset.
* 16/06/2021: Task adaptive learning following this paper https://arxiv.org/pdf/2004.10964.pdf and hugging face notebook https://github.com/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb. Interesting paper introducing differential learning rate: https://arxiv.org/pdf/1801.06146.pdf
* 19/06/2021 - 20/06/2021: Read the folling paper about how to fine tune bert https://arxiv.org/pdf/1905.05583.pdf. Trained a simple MLP model. Add hidden layer of the MLP to BERT and creater an inference pipeline. 
* 21/06/2021: Finished the MLP and bert + MLP hidden layer. Submission tomorrow.
* 22/06/2021: Read the paper revisiting few sample bert fine tuning https://arxiv.org/pdf/2006.05987.pdf. Problem with kaggle API Bad request error 404 ... I can not upload my models to kaggle. Will retry tomorrow. It was working yesterday.
* 23/06/2021: TO DO: read the following paper about weight initialization techniques for pretrained LM: https://arxiv.org/pdf/2002.06305.pdf + read the following paper about optimizing deeper transformer on small datasets: https://arxiv.org/pdf/2012.15355.pdf
* 24/06/2021: learned about pytorch data samplers and sequence bucketing using the following notebook: https://www.kaggle.com/shahules/guide-pytorch-data-samplers-sequence-bucketing
* 25/06/2021: there was a problem with my roberta+MLP model ... I am still working on it.
* 26/06/2021: first inference with roberta + MLP gave 0.508 in leaderboard.
* 27/06/2021: retrain the model with a new architecture and training pipeline. Inference CV: 0.43 LB: 0.487. Read the following papers about adding numerical features to bert: https://arxiv.org/pdf/2106.07935.pdf + https://arxiv.org/pdf/2103.04083v1.pdf (hierarchical transformers) + bonne vidéo introductive sur le mécanisme d'attention https://www.youtube.com/watch?v=LALTmQhVkfU.
* Single Roberta model cv 0.472 LB 0.481
* 28/06/2021: Found a new channel discussing about bert (very interesting): https://www.youtube.com/c/ChrisMcCormickAI/videos. Great overview of Bert: https://www.youtube.com/watch?v=TQQlZhbC5ps&t=638s. Read a second time the blog http://jalammar.github.io/illustrated-transformer/ explaining transformers.
* 01/07/2021:  https://www.youtube.com/watch?v=x66kkDnbzi4&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6&index=3 + study the following notebook which introduces attention head https://www.kaggle.com/andretugan/pre-trained-roberta-solution-in-pytorch
* 02/07:2021: https://www.youtube.com/watch?v=C4jmYHLLG3A&t=2s + Trying to understand why cv decreases but not lb.
* 
## Things to try


- THIS WEEKEND ADD CUSTOM MLP LAYER TO BERT Done
- Change the loss function
- Multi Sample Dropout not working well
- Mean Max head: not working well
- try convolutional heads
- try attention head
- try pseudo labeling as the dataset is very small
- More regularization techniques like batchnorm, layer norm work well
- pretrain bert on our corpus Work well
- Validate multiple times in one epoch Work well
- Use Distillbert (smaller model)
- character level bert / sentence level bert
- Check the impact of the seed https://www.kaggle.com/khoongweihao/tse2020-roberta-cnn-random-seed-distribution
- try to add custom features: https://www.kaggle.com/ravishah1/readability-feature-engineering-non-nn-baseline / https://arxiv.org/pdf/2010.09078.pdf
- Use simple LSTM then concat with mlp hidden layer (no one do it but good results in two papers about it. I will try it after tuning the Bert Model).
- try data augmtentation: from english to french, from french to english

## Papers to look at

* About bert stability: https://arxiv.org/pdf/2006.04884.pdf

## Notebooks to look at

* https://www.kaggle.com/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning?scriptVersionId=65609052

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
