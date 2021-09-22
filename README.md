# CommonLit Readability Prize Competition (Kaggle): Competition and Project Overview
**Final product hosted on Heroku:** https://text-comparator.herokuapp.com/

**Faire un menu deroulant

## Competition Overview

* The goal of the competition is to create machine learning models to evaluate the reading level of passages of text. Being able to rate the complexity of a text is important for improving students reading capabilities. Indeed, studies show that students can make faster progress in learning if they are taught with texts just above their reading capabilities (“https://mydigitalpublication.com/publication/?m=13959&i=644729&p=18&ver=html5”). Assessing the level of a text has always been a challenging task and a lot of methods have been developed in order to provide students with appropriate texts. These methods are based on “weak proxies of text decoding (i.e., characters or syllables per word) and syntactic complexity (i.e., number or words per sentence)” (https://www.kaggle.com/c/commonlitreadabilityprize). The final goal is to create machine learning models that could be used by literacy curriculum developers and teachers to choose the best passages for their classrooms. Also, with this kind of algorithms, students could receive feedbacks on their complexity and readability of their work.
* In this competition, I finished in the top 18% (I had a submission for the top 8% but I did not choose it as my final submission) over 3633 teams. In my opinion, there was five main points of succes in the competition:
- Stabilize the training of transformers: the targets were very noisy 
- Evaluate the performance of our models within each epochs and not just after each epochs
- Setting Dropout parameters of the transformers model to 0
- Finding appropriate datasets for pseudo labeling: the training set is very small
- Create a good ensemble: we had 3 hours to make predictions
* We will cover all that points later in this documentation.

## Project Overview

* I love Kaggle Competitions. This is a great way to learn a lot of things regarding the modeling part of a data science project. You can learn from an incredible community and you have also a feedback of what you have done compared to the others with the leaderboard. But kaggle misses some parts of a data science project like for example collecting the data and deploying the model in production. With respect to this last point, I decided to deploy one of my models and even to incorporate different techniques from the winners of the competition (web app link : https://text-comparator.herokuapp.com/). The goal of the web app is to compare the reading level of two different texts. Why is it useful to compare the reading level of two texts ? Have you ever been in a situation where you wanted to explain a complex subject to someone but too many definitions came to your mind so that you could not the most suitable one ? This web app will help you choose the most suitable definition to give to a novice. Under the hood, a bert model assesses the reading level of the two texts and then compare them and returns the most easy one (the model that I deployed is not the best one because of model size constarints for deployments.
* I also tried to reproduce the best techniques that the winners of the Competition have used. At the end, it improved my model a lot.

## Path to final submission

#### What worked for me

* layer wise learning rate / decay learning rate
* large models roberta large and electra large
* Mean/ Attention Pooling
* Gradient accumulation: Difficult to train large transformer model with a batch size higher than 8 on colab. The problem is, the lower the batch size, the higher the impact of noise on training, thus the use of gradient accumulation to simulate larger batch size (and remove the noise)
* Multi Sample Dropout (idea come from this paper: https://arxiv.org/pdf/1905.09788.pdf)
* Stochastic Weight Averaging (SWA)


#### What did not work for me

* adding numerical features to transformers
* Reinitialize last layers of bert model (the idea come ffrom this paper: https://arxiv.org/pdf/2006.05987.pdf
* mask words randomly during training
* smart batching
* Mixout (the idea come from this paper)

* Before diving into the training of huge transformer models, I wanted to experiments wth simpler ones in order to create a baseline. At first, I generated different features based on usual readability formula and then trained a gradient boosting model on this generated dataset. I then switched to RNN, but they seemed to perform worse than gradient boosting and feature engineering models. Thus, I started to use transformers and learn about the hugging face library. 
	Path to training bert models.

#### Story

* At first, the training was totally unstable. After reading the paper (paper name: https://arxiv.org/pdf/1801.06146.pdf) and after looking at the following notebook https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-fit?scriptVersionId=64693127, I started to use layer wise learning rate. Layer wise learning rate and dropout removal enabled to stabilize the training of bert. Put a graph from wandb to show that. The problem with the instability of bert like models is very well known and is especially important for small datasets (like the one we were working with : less than 3000 training samples). Many papers tried to tackle this problem (https://arxiv.org/pdf/2006.05987.pdf +  https://arxiv.org/pdf/1905.05583.pdf + https://arxiv.org/pdf/2012.15355.pdf). 
(Show both dropout removal and layer wise learning rate in code).
After finding a way to stabilize the training of bert, I then tried to add numerical features to it following these papers advices ( https://arxiv.org/pdf/2106.07935.pdf + https://arxiv.org/pdf/2103.04083v1.pdf +  https://aclanthology.org/2021.maiworkshop-1.10.pdf). It improved my cv, but it seems that it was not generalizing to the public test set. I decided to give up with this idea.

* I then tried different head for bert and tried to use them on different outputs from the encoder (pooler output, last hidden state, hidden states). In order to perform the best analysis as possible I ran all my experiments with 3 different seeds because of the randomness of transformers training. At the end, for my set up, it seems that the attention head and the mean pooling performed the best. I decided to keep focusing on these two heads for the rest of the competition.
(Show the head in code)
I then tried Stochastic Weight Averaging (SWA) that improved a bit my CV and LB score. Layer reinitialization was promising but did not work for my set up.
As many kagglers where talking about the importance of  ensembles, I looked at different way to decrease inference time. I used Smart Batching for inference but for training it hurts oo much the performance (it was still a great way to experiment faster).
At the end, my final submission was just a blending of Roberta large with mean pooling, Roberta large with attention pooling, Roberta base with attention pooling and electra large with mean pooling (you can find the notebook in (path for the notebook for inference). Some of them where trained with layer wise learing rate and some with decay learning rate.

## What I learned from best solutions

After studying the best performing solutions of the competition, I started to incorporate some of the tricks that I have learned to my own models. 
* It seems that one of the key to success was to use different transformer models (to introduce as much diversity as possible) and to make an ensemble of them (the larger models performed way better than the smaller ones). For my own solution, I fine tuned two type of models: roberta and electra. To improve my solution I decided to add to my ensemble funnel large and deberta large.
* The second main improvement that I have made to my models after understanding the best solutions is the use of pseudo labeling. As the training set was very small, increasing its size was crucial. The problem with pseudo labeling is that we need to find external data that are representative of the training data. The winner of the competition introduced a novel technique to do so: create a large external dataset that seemed relevant for the competition, then make text snippets of the external data that have approximately the same length as the training data. Then use sentence bert "to generate sentence embeddings and retrieve the five text snippets which had the highest cosine similarity to the original excerpt" (https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257844). I tried this idea by myself on my electra model(you can find my code in the Notebooks folder) and it gave me a huge improvement of 0.01.

## How to use this repository

## Code and Resources Used

**Python Version:** 3.8

**For Web Framework Requirements:** ```pip install -r requirements.txt```

**Flask Productionization:** https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/

**Transformer outputs:** https://www.kaggle.com/rhtsingh/utilizing-transformer-representations-efficiently

**Stochastic Weight Averaging:** https://www.kaggle.com/rhtsingh/swa-apex-amp-interpreting-transformers-in-torch

**Stability of Transformer fine tuning:** https://www.kaggle.com/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning

**Smart Batching:** https://www.kaggle.com/rhtsingh/speeding-up-transformer-w-optimization-strategies
 
## Best ressources for NLP


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
* 30/07/2021: Read the following paper about unsupervised data augmentation https://arxiv.org/pdf/1904.12848.pdf. Do not have time to implement it but will do it after the competition ends. Tried to mask words randomly during training but it seems that it hurts training performance. Completely understood how to finetune roberta large for this dataset (no dropout but layer norm + attention or mean head. The learning rate is very important. Decay learning rate works way better than layer wise elarning rate !).
* 01/08/2021: Creating blending scripts with optuna + training funnel models (paper: https://arxiv.org/abs/2006.03236)
* Top 10% submission. Create another Roberta with Max Len 300 and tried again MLP + Roberta (did not work !)
* 02/08/2021: Keep trying to work on my blending. Learned about Deberta https://arxiv.org/abs/2006.03654. Trying to finetune it.
* 31/08/2021: Started to look at top solutions
* 11-12/09/2021: understanding of sentence berte https://arxiv.org/pdf/1908.10084.pdf
* Very interesting paper about self training https://arxiv.org/format/2010.02194
* 13/09/2021: interesting article about semantic search + creation of the script to get external data



* Try pseudo labeling with std. I need to try combining std and targets! MTL ??


* IDEA / TRAIN THE MODEL ON CLASSIFICATION THEN REGRESSION TASK or combine two bert models one on classification and the other on regression task.

* Found a very interesting blog about NLP: https://mccormickml.com/
* Add directly the numerical features then concat the attention layer or multi headed self attention layer check https://aclanthology.org/2021.maiworkshop-1.10.pdf github. Test the different possibilities. But tomorrow first understand and finetune the following notebook : https://www.kaggle.com/andretugan/lightweight-roberta-solution-in-pytorch

## Best solutions

* Different models : roberta large, electra large, deberta large, luke large, funnel large
* key parameters: lerning rate, head, dropout removal, pseudo labeling, activation, loss function
* Importance of model diversity
* Make sure that external datasets are representative of the training data
* try a bigger k in kfold cross val to have a better cross validation strategy

* Google sheet link for experiments https://docs.google.com/spreadsheets/d/1zMy1EWSGylj0xPF-PY_FLF2q1VUcLISbPF_MRbO-xGw/edit?usp=sharing

### From 1100 to 40 th in 3 days
* https://www.youtube.com/watch?v=XLaq-boyAGk&t=10s
* https://github.com/VigneshBaskar/kaggle_commonlit/blob/main/solution_walkthrough.ipynb
* https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258363
* Pretraining using wikipedia data using rankingloss
* Mask added attention head
* Layer wise learning rate

### 6th place solution (Gaussian process regression (GPR))
* https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258554
* Pseudo labelling + GPR
* Change pretrained models to squad and triviaqa
* Dropout Removal

### Private 22nd Place solution
* gradient clipping 
* reinitialize last layers
* LSTM heads

### 9th place solution -Ensemble of four models
* cross validation strategy: increase the number of folds to 15.
* use deberta base to find best params for deberta large. use this paper: https://arxiv.org/pdf/2006.04884.pdf
* BCEWithLogitsLoss as a loss function "we trained BCEWithLogitsLoss using Sigmoid(target-Median(target)) as the prediction label. Median(target) was added to the regressor during prediction"

### 15th Place
* https://www.kaggle.com/c/commonlitreadabilityprize/discussion/260800
* SVR head

### 1st place solution - external data, teacher/student and sentence transformers
* https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257844
* "made a large collection of external data. I used sentence bert (Reimers and Gurevych 2019 - https://github.com/UKPLab/sentence-transformers) to select for each excerpt in the train set the 5 most similar snippets in the external data collection"
* trained some different models on the pseudo-labeled data for 1-4 epochs. After training on the pseudo-labeled data, I trained each model on the original training set. 
* Use external data from simplewiki, wikipedia and book corpus
* Sentence Bert paper: https://arxiv.org/abs/1908.10084
* https://www.sbert.net/
* https://github.com/mathislucka/kaggle_clrp_1st_place_solution


### Single PairWise model
* https://www.kaggle.com/c/commonlitreadabilityprize/discussion/258283
* Interesting idea to try.

### An Alternative Approach - Compare Excerpts
* https://www.kaggle.com/c/commonlitreadabilityprize/discussion/257446
* Interesting idea to try also for pretraining for example

## Project Idea

You want to explain a topic to someone (kids, novice, expert), you have multiple definitions that comes to your mind, using the api decide the definition you should chose. First the user enter for what type of person he wants to give this definition. Then, enters 2 to 3 diffzerent definitions. After that, the model will select the most suitable one for the usecase. Deploy on AWS or Heroku or maybe on tensorflow lite.

## Proposition for the PFE

* weights and biases dashboard: https://wandb.ai/ruchi798/commonlit?workspace=user-ruchi798
* Find external datasets (sentence bert) (https://www.kaggle.com/markwijkhuizen/discussion)
* Improve models using pseudo labeling 
* Try to use differential learnign rate/ bigger k/ remove all dropout
* Create a script to reset training if it diverges (example seed/ mlm or not)
* Try new loss functions / Different heads
* Try single pairwise model
* Train at least 2 different single models like electra deberta funnel or roberta
* Deploy the web app and create an article on medium explaining how to turn a kaggle competitions into an application (explain everyhting about what we used to create the model and the deployment part).

## Weight and Biases learning

* https://theaisummer.com/weights-and-biases-tutorial/
* https://towardsdatascience.com/machine-learning-experiment-tracking-93b796e501b0


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
