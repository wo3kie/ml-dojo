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
  
### binary_cross_entropy.ipynb  
Demonstrate how to implement the `binary cross-entropy` loss function with a custom autograd backward function in PyTorch.
  
### cross_entropy.ipynb  
Demonstrate how to implement the cross-entropy loss function with a custom autograd backward function in PyTorch.
  
### common.ipynb  
Contains common utilities and helper functions used across the notebooks.  
  
### entropy.ipynb  
Explore the concept of entropy, its properties, and how it can be calculated for different distributions.  
   
### gradient.ipynb  
Explore the concept of gradient for scalar functions.
  
### inner_product.ipynb  
Explore the concept of inner product for vectors and matrices, and its properties.
  
### linear.ipynb  
Explore the concept of linear function with its backward method and linear module.
  
### newton.ipynb  
Demonstrate how to use Newton's method to find extrema of a single-variable/multivariate function using first and second derivatives.  
    
### per_lin_sig_bce.ipynb  
Demonstrate how to implement a perceptron using a `Linear` to the affine transform, a `Sigmoid` activation for binary classification, and a `BinaryCrossEntropy` loss for training. Present three variants, high-level full PyTorch implementation using build-in autograd, a mid-level version with custom `autograd.Function`, and a fully manual implementation using hand-derived analytical gradients.
    
### probability.ipynb  
Demonstrate how to calculate probabilities and conditional probabilities.  
  
### sigmoid.ipynb  
Explore the sigmoid function, its properties, and implement its overflow resistant version.  
  
### tanh.ipynb  
Explore the hyperbolic tangent function, its properties, and implement its overflow resistant version.  
    