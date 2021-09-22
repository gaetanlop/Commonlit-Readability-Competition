# CommonLit Readability Prize Competition (Kaggle): Competition and Project Overview
**Final product hosted on Heroku:** https://text-comparator.herokuapp.com/

**Faire un menu deroulant

## Competition Overview

* The goal of the competition is to create machine learning models to evaluate the reading level of passages of text. Being able to rate the complexity of a text is important for improving students reading capabilities. Indeed, studies show that students can make faster progress in learning if they are taught with texts just above their reading capabilities (“https://mydigitalpublication.com/publication/?m=13959&i=644729&p=18&ver=html5”). Assessing the level of a text has always been a challenging task and a lot of methods have been developed in order to provide students with appropriate texts. These methods are based on (https://www.kaggle.com/c/commonlitreadabilityprize)
> weak proxies of text decoding (i.e., characters or syllables per word) and syntactic complexity (i.e., number or words per sentence).

The final goal is to create machine learning models that could be used by literacy curriculum developers and teachers to choose the best passages for their classrooms. Also, with this kind of algorithms, students could receive feedbacks on their complexity and readability of their work.
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
* Training the model on classification task then on regression task

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

This github repo is composed of two parts:

#### Notebooks folder 
- In this folder, you can find some of my notebooks that I developed during and after the competition (when I incorporated tricks from top solutions.
* Blending-Inference-Kaggle : It is the notebook that I used to make inference on kaggle.
* Blending-Weights-Optuna : In this notebook, I used Optuna to find the best weights for each of my model for the final submission on Kaggle
* Create-External-Data : In this notebook, I gathered some external data following the tricks of the competition winner.
* Deberta_Large_Training : This is one of my notebook to train a large transformer model from hugging face.
* Pseudo Labeling : In this notebook, I used my best models fromt he competition to pseudo label the selected external data.

#### Other folder and files 
The other folder and files are all related to deployment.
* app.py : the main app
* torch_utils.py : some useful pytorch functions that I used in app.py
* Templates folder : HTML code
* Static folder : css code

## Code and Resources Used

**Python Version:** 3.8

**For Web Framework Requirements:** ```pip install -r requirements.txt```

**Flask Productionization:** https://www.python-engineer.com/posts/pytorch-model-deployment-with-flask/

**Transformer outputs:** https://www.kaggle.com/rhtsingh/utilizing-transformer-representations-efficiently

**Stochastic Weight Averaging:** https://www.kaggle.com/rhtsingh/swa-apex-amp-interpreting-transformers-in-torch

**Stability of Transformer fine tuning:** https://www.kaggle.com/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning

**Smart Batching:** https://www.kaggle.com/rhtsingh/speeding-up-transformer-w-optimization-strategies
 

