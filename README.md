# ai.c

Artificial Neural Network library in C. Made in 10 days as a submission to [Code Guessing #57](https://cg.esolangs.gay/57).
The files I've submitted can be viewed in *code_guessing* branch.

The dataset I've used to train my network is a modified version of [this](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) dataset.

# Details
This library implements dense and dropout(\*) layers, SGD algorithm,
four activation functions, mean squared error and categorical cross entropy loss functions. 

\* The dropout layer doesn't really work, so I've temporarily
added a parameter to dense layers. I might update it in the future

# FAQ
### What was your goal when creating this?
I wanted to grasp how ANNs really worked, I've already wanted to implement something like this and Code Guessing \#57 provided a good opportunity for me to do so.

### Why C?
I wanted it to be more challenging.
