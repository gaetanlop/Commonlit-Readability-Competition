# Commonlit-Readability

## Interesting notebooks:

* About seeds and cnn heads: https://www.kaggle.com/khoongweihao/tse2020-roberta-cnn-random-seed-distribution

## Results:

* Random seeds make training unstable. We are unable to get the same result a second time if we are using a different random_state

## Things to try

- Check the impact of the seed 
- Choose one seed and check if using the best number of epochs gives the best results. Rerain on all the dataset.
- Check if stacking last hidden states of concat last hidden states with linear or conv1d head improves performance.
- Choose a good sentence length
