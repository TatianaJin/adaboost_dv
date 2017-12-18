# Adaboost Dynamic Votes

Pattern Recognition course project

## Dependency
* python 2.7
* scikit-learn library

## Implementations
1. basic adaboost benchmark (binary)
2. adaboost with votes adjusted after all base classifiers have been selected
3. adaboost with votes adjusted at the end of each iteration

# Experiments
## Datasets (from LIBSVM datasets)
* a9a
* breast-cancer
* gisette

## Performance
Limit the maximum number of weak classifiers, using Decision Tree as the weak classifier.
* training accuracy
* testing accuracy
* training time

## Contribution
1. A method to update the votes, exploiting the tradeoff between accuracy and training time
2. The update rule of sample weights
3. Extensive experimental evaluation
