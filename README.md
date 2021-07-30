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
* 04/07/2021: https://www.youtube.com/watch?v=YIEe7d7YqaU + https://www.youtube.com/watch?v=0U1irnILcN0 about bert inner workings. https://aclanthology.org/2021.maiworkshop-1.10.pdf paper about learning tabular and text data with transformers.
* 05/07/2021: followed the following notebook to undertsnad optimizers: https://www.kaggle.com/andretugan/lightweight-roberta-solution-in-pytorch
* 07/07/2021: implemented layer wise learning rate
* 13/07/2021: Tried layer wise learning rate + freezing + read the paper antoher time https://arxiv.org/pdf/1905.05583.pdf Layer wise lr not adding much. Try freezing only some layers of roberta.
* 19/07/2021: find similarities between text but not the same score. Need to find a way to deal with this noisy samples.
* 20/07/2021: Leaned about bradley terry algo. Tried to remove outliers and change the loss function to weight each samples differently based on its standard error. It did not wrok. Currently working on a new wqy to pretrain a model with bins. Need to try xgb or svm on CLS token + numerical features.
* 21/07/2021: Tried to binarize the target variable and train on bert model on a classification task. Use this model as a pretrained model. Currently under testing.
* https://www.kaggle.com/rhtsingh/utilizing-transformer-representations-efficiently learning from this notebook different head for bert.
* 22/07/2021: Bert paper https://arxiv.org/pdf/1810.04805.pdf + understanding feature extraction with bert https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/.
* Learn about layer reinitialization and implement it (https://arxiv.org/pdf/2006.05987.pdf) + learned about mixout (https://arxiv.org/pdf/1909.11299.pdf) and implement it. Both techniques yields to better results on cv ! 
* 23/07/2021: Learned SWA and implement it https://arxiv.org/pdf/1803.05407.pdf + https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
* 24/07/2021: Understand how to get correlation between lb and cv by adding mroe regularization. Leads to worst cv but better correlation between lb and cv.
* 25/07/2021: Trying to understand if adding an MLP trained on numerical features could help base roberta. 0.491 for unique Roberta and 0.58 for the MLP on LB. 
* 26/07/2021: Read the following paper about bert instability https://arxiv.org/pdf/2006.04884.pdf. Ways to solve: increase the number of training iterations even if 0 loss.
* 27/07/2021: Smart Batching understanding + Implementation (https://www.kaggle.com/rhtsingh/speeding-up-transformer-w-optimization-strategies) + Undertsanding MTL + implementation (https://ruder.io/multi-task/ + https://www.kaggle.com/c/commonlitreadabilityprize/discussion/238384). MTL implementation did not improve CV ... + Created a simple baseline for roberta Large. + Read Electra paper https://arxiv.org/pdf/2003.10555.pdf
* 28/07/2021: Undertsanding the following blog post about samplers, dataloader, collators https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html#SequentialSampler. + Finish smart batching implementation (leads to worst cv and lb scores ...). Roberta Large perform better than roberta base on lb but not on cv. Read the following paper comparing fine tuning to feature extractor https://arxiv.org/pdf/1903.05987.pdf. Read the paper about seeds https://arxiv.org/pdf/2002.06305.pdf then created a notebook to find the best seeds ! 
* 30/07/2021: Read the following paper about unsupervised data augmentation https://arxiv.org/pdf/1904.12848.pdf. Do not have time to implement it but will do it after the competition ends. Tried to mask words randomly during training but it seems that it hurts training performance.
* 



* Try pseudo labeling with std. I need to try combining std and targets! MTL ??


* IDEA / TRAIN THE MODEL ON CLASSIFICATION THEN REGRESSION TASK or combine two bert models one on classification and the other on regression task.

* Found a very interesting blog about NLP: https://mccormickml.com/
* Add directly the numerical features then concat the attention layer or multi headed self attention layer check https://aclanthology.org/2021.maiworkshop-1.10.pdf github. Test the different possibilities. But tomorrow first understand and finetune the following notebook : https://www.kaggle.com/andretugan/lightweight-roberta-solution-in-pytorch

## Tomorrow

* Build a pipeline to test different seeds

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
