## Copyright (C) 2026 Łukasz Czerwiński

![ML Dojo](ml-dojo-200x300.png)

# ML-dojo

### Do you want to learn how to _learn_?

A collection of building blocks for understanding neural networks internals.
  
Each component, like _linear_, _sigmoid_, _tanh_, _relu_, _bce_... is implemented with explicit _forward_ and _backward_ passes. All derivatives are derived manually. These primitives are then composed into progressively richer perceptrons: from the simplest linear models to fully nonlinear architectures. Everything is written in PyTorch, but with deliberate exposure of the underlying mathematics.

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
  
### approx.ipynb  
Demonstrate how to implement an `approx` class for approximate equality comparison of floating-point numbers, with support for both absolute and relative tolerances, and customizable logging of failed comparisons.

```engine=python
assert 1.05 == approx(1.0, atol=0.1)
assert 1.05 == approx(1.0, rtol=0.1)
```

### backward.ipynb  
Demonstrate how to implement a custom autograd `backward` function in PyTorch on some examples.  
  
### bce.ipynb  
Demonstrate how to implement the `binary cross-entropy` loss function with a custom autograd backward function in PyTorch.
  
### cross_entropy.ipynb  
Demonstrate how to implement the cross-entropy loss function with a custom autograd backward function in PyTorch.
  
### common.ipynb  
Contains common utilities and helper functions used acro_ss the notebooks.  
  
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
    
### per_lin_tanh_bce.ipynb  
Demonstrate how to implement a perceptron using a `Linear` to the affine transform, a `Tanh` activation for binary classification, and a `BinaryCrossEntropy` loss for training. Present three variants, high-level full PyTorch implementation using build-in autograd, a mid-level version with custom `autograd.Function`, and a fully manual implementation using hand-derived analytical gradients. Compare learning performance of the `Tanh` activation with the `Sigmoid` activation.
  
### probability.ipynb  
Demonstrate how to calculate probabilities and conditional probabilities.  
  
### sigmoid.ipynb  
Explore the sigmoid function, its properties, and implement its overflow resistant version.  
  
### sign.ipynb  
Implement the sign function, with its custom autograd backward function in PyTorch.

### tanh.ipynb  
Explore the hyperbolic tangent function, its properties, and implement its overflow resistant version.  
    