## Copyright (C) 2026 Łukasz Czerwiński

# ML-dojo

Small collection of notebooks exploring Machine Learning using PyTorch library.

## Website
https://github.com/wo3kie/ml-dojo


## Setup

### Create a fresh venv

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run the notebooks
  
Open any of the notebooks listed above in the browser.  
  
## Contents  
  
### common.ipynb  
Contains common utilities and helper functions used across the notebooks.  
  
### cross_entropy.ipynb  
Explore the concept of cross-entropy and how it relates to entropy and optimal coding.
  
### entropy.ipynb  
Explore the concept of entropy, its properties, and how it can be calculated for different distributions.  
  
### functions.ipynb  
Explore various PyTorch functions.  
  
### lin_reg_1D_closed_form.ipynb  
Demonstrate how to solve linear regression in one dimension using the closed-form solution.
  
### lin_reg_1D_normal_equation.ipynb  
Demonstrate how to solve linear regression in one dimension using the normal equation.
  
### lin_reg_ND_sgd_gradient.ipynb  
Demonstrate how to solve linear regression in many dimensions using stochastic gradient descent (SGD),
and derive the gradients for weights and bias manually.  
  
### lin_reg_ND_sgd_autograd.ipynb
Demonstrate how to solve linear regression in many dimensions using stochastic gradient descent (SGD),
and use PyTorch's autograd to compute the gradients for weights and bias.
  
### log_reg_2C_sgd_autograd.ipynb  
Demonstrate how to solve logistic regression for two classes using stochastic gradient descent (SGD),
and use PyTorch's autograd to compute the gradients for weights and bias.
  
### log_reg_2C_sgd_gradient.ipynb  
Demonstrate how to solve logistic regression for two classes using stochastic gradient descent (SGD),
and derive the gradients for weights and bias manually.  
  
### log_reg_NC_sgd_autograd.ipynb  
Demonstrate how to solve logistic regression for multiple classes using stochastic gradient descent (SGD),
and use PyTorch's autograd to compute the gradients for weights and bias.
  
### mnist_log_reg_NC_sgd.ipynb  
Demonstrate how to implement the Image Recognition task for the MNIST dataset using logistic regression for 
multiple classes with stochastic gradient descent (SGD), performing all gradients calculations step-by-step manually.  
  
### mnist_log_reg_NC_sgd_weights.ipynb  
Demonstrate how to implement the Image Recognition task for the MNIST dataset using logistic regression for 
multiple classes with stochastic gradient descent (SGD), and visualize the learned weights as images for each digit.
  
### newton_1D.ipynb  
Demonstrate how to use Newton's method to find extrema of a single-variable function using first and second derivatives.  
  
### newton_ND.ipynb  
Demonstrate how to use Newton's method to find extrema of a multivariate function using gradient and Hessian.  
  
### probability.ipynb  
Demonstrate how to calculate probabilities and conditional probabilities.  
  
### sigmoid.ipynb  
Explore the sigmoid function, its properties, and implement its overflow resistant version.  
  
### tensor.ipynb  
Demonstrate how to create and manipulate tensors in PyTorch, including basic operations, indexing, and reshaping.  
  