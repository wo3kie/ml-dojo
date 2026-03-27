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
  
### backward.ipynb  
Demonstrate how to implement a custom autograd `backward` function in PyTorch on some examples.  
  
### cross_entropy.ipynb / binary_cross_entropy.ipynb  
Demonstrate how to implement the cross-entropy / binary cross-entropy loss function with a custom autograd `backward` function in PyTorch.
  
### common.ipynb  
Contains common utilities and helper functions used across the notebooks.  
  
### entropy.ipynb  
Explore the concept of entropy, its properties, and how it can be calculated for different distributions.  
  
### functions.ipynb  
Explore various PyTorch functions.  
  
### inner_product.ipynb  
Explore the concept of inner product for vectors and matrices, and its properties.
  
### linear.ipynb  
Explore the concept of linear function with its backward method and linear module.
  
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
  
### log_reg_2C_sgd_autograd.ipynb / log_reg_NC_sgd_autograd.ipynb   
Demonstrate how to solve logistic regression for two/multiple classes using stochastic gradient descent (SGD),
and use PyTorch's autograd to compute the gradients for weights and bias.
  
### log_reg_2C_sgd_gradient.ipynb / log_reg_NC_sgd_gradient.ipynb  
Demonstrate how to solve logistic regression for two/multiple classes using stochastic gradient descent (SGD),
and derive the gradients for weights and bias manually.  
    
### mnist_log_reg_NC_sgd.ipynb  
Demonstrate how to implement the Image Recognition task for the MNIST dataset using logistic regression for 
multiple classes with stochastic gradient descent (SGD), performing all gradients calculations step-by-step manually.  
  
### mnist_log_reg_NC_sgd_demo.ipynb  
Demonstrate how to implement the Image Recognition task for the MNIST dataset using logistic regression for 
multiple classes with stochastic gradient descent (SGD), and visualize the learned weights as images for each digit.
  
### newton_1D.ipynb / newton_ND.ipynb   
Demonstrate how to use Newton's method to find extrema of a single-variable/multivariate function using first and second derivatives.  
    
### per_lin_sig_bce_gradient.ipynb / per_lin_sig_bce_backward.ipynb  
Demonstrate how to implement a perceptron using a linear model followed by a sigmoid for binary classification. Derive all `gradient`/`backward` functions manually. Test the implementation on boolean functions like AND, OR, NAND and XOR.
  
### per_lin_tanh_bce_gradient.ipynb / per_lin_tanh_bce_backward.ipynb  
Demonstrate how to implement a perceptron using a linear model followed by a hyperbolic tangent for binary classification. Derive all `gradient`/`backward` functions manually. Test the implementation on boolean functions like AND, OR, NAND and XOR.
  
### per_lin_sig_lin_sig_bce.ipynb  
Demonstrate how to implement a two-layer perceptron using linear models followed by sigmoid activations for binary classification. Derive all `gradient` functions manually. Test the implementation on boolean functions like AND, OR, NAND and XOR.  
  
### perceptron_log_demo.ipynb  
Demonstrate how to use the logistic perceptron implementation to learn boolean functions like AND, OR, NAND and XOR, and visualize the decision boundaries.
  
### probability.ipynb  
Demonstrate how to calculate probabilities and conditional probabilities.  
  
### sigmoid.ipynb / tanh.ipynb  
Explore the sigmoid/tanh function, its properties, and implement its overflow resistant version.  
  
### tensor.ipynb  
Demonstrate how to create and manipulate tensors in PyTorch, including basic operations, indexing, and reshaping.  
  