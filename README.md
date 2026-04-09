## Copyright (C) 2026 Łukasz Czerwiński

![ML Dojo](ml-dojo-200x300.png)

# ML-dojo

## Do you want to learn how to _learn_?

A curated collection of building blocks for understanding the internals of neural networks.

Each component — such as linear, _sigmoid_, _tanh_, _relu_, _bce_ — is implemented with explicit forward and backward passes. All derivatives are derived manually. These primitives are then composed into progressively richer perceptrons: from the simplest linear models to fully nonlinear architectures. Everything is written in PyTorch, with deliberate exposure of the underlying mathematics.

## Website

https://github.com/wo3kie/ml-dojo

## Setup

## Create a fresh venv

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run the notebooks
  
Open any of the notebooks listed above in the browser.  
  
## Contents  
  
## approx.ipynb  
Implement an `approx` utility for approximate equality comparison of floating-point numbers, supporting both absolute and relative tolerances, with optional logging of failed comparisons.

```engine=python
assert 1.05 == approx(1.0, atol=0.1)
assert 1.05 == approx(1.0, rtol=0.1)
```

## backward.ipynb  
Explores how PyTorch’s _backward_ method defines gradient flow through custom operations. The notebook shows how to manually implement a backward pass that receives the upstream gradient and returns gradients with respect to each input.
  
## bce.ipynb  
Binary cross‑entropy measures the divergence between a predicted probability 𝑝 and a true binary label 𝑡 ∈ {0, 1}, penalizing confident but incorrect predictions. The notebook implements BCE with a custom autograd backward function.
    
## common.ipynb  
Utility functions and helpers shared across notebooks.  
  
## cross_entropy.ipynb  
Cross‑entropy generalizes binary cross‑entropy to multi‑class classification. It measures the divergence between a predicted probability distribution 𝑝 and a true one‑hot label 𝑡.
  
## entropy.ipynb  
Introduces entropy as a measure of uncertainty in a probability distribution. The notebook computes entropy for geometric, uniform, and Zipf‑like distributions, illustrating how uncertainty changes with the shape of the probability mass.
   
## gradient.ipynb  
Defines the gradient of a scalar function with respect to vector or matrix inputs. For vectors, the gradient is a vector of partial derivatives; for matrices, it is a matrix of the same shape.
The differential 𝑑𝑓 is introduced as the linear approximation of the function’s change, expressed as the inner product between the gradient and the perturbation.
  
## inner_product.ipynb  
Introduces the inner product for vectors and generalizes it to matrices via the Frobenius inner product. The notebook also explains why transposition appears naturally in gradient formulas: the inner‑product structure forces gradients to align with the adjoint (transpose) of the corresponding linear map.
  
## linear.ipynb  
Presents the fundamental linear layer 𝑧=𝑥𝑊+𝑏. This affine transformation combines a linear map with a translation, preserving straight lines and parallelism while shifting decision boundaries. Linear layers form the backbone of neural networks across all architectures.
  
## mse.ipynb  
Mean squared error measures the average squared difference between predictions and targets. It penalizes large deviations strongly and corresponds to maximum‑likelihood estimation under Gaussian noise. The notebook implements MSE with a custom backward pass. 
  
## newton.ipynb  
Newton’s method finds a minimum (or root) of a scalar function by taking a second‑order Taylor expansion around the current point and solving for where the linearized derivative becomes zero. 
  
$$ f(x) \approx f(x_n) + f'(x_n)(x - x_n) + \frac{1}{2}f''(x_n)(x - x_n)^2 $$
  
For one-dimensional version it iteratively updates the guess using the formula:  
  
$$ x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} $$
    
In multiple dimensions, the first derivative generalizes to the gradient, and the second derivative generalizes to the Hessian matrix. The update becomes:

$$\mathbf{x}_{n+1} = \mathbf{x}_n - H^{-1} \nabla f(\mathbf{x}_n)$$
  
This method converges quadratically near a local minimum, making it very efficient when the function is well‑behaved and the initial guess is close to the solution. In this notebook, we will implement Newton's method for finding minima of scalar functions.  
  
## neuron.ipynb  
Describes the McCulloch–Pitts neuron (1943), a logical threshold unit using a hard step activation. It implements Boolean functions but cannot learn XOR due to linear inseparability.
  
## per_lin_mse.ipynb  
Implements a linear perceptron trained with mean squared error — historically known as ADALINE (ADAptive LInear NEuron), introduced by Widrow and Hoff in 1960. ADALINE was a major milestone in the development of artificial intelligence: the first model to use a continuous output and to update its weights using a true gradient of a differentiable loss function, decades before backpropagation was formalized.
    
## per_lin_ploss.ipynb  
Implements Rosenblatt’s perceptron — the first trainable artificial neuron. Learning is performed via the perceptron update rule, equivalent to a subgradient step on the perceptron loss.
  
## per_lin_sig_bce.ipynb  
Implements a perceptron using `Linear → Sigmoid → BCE`. Three variants are shown: full PyTorch autograd, custom autograd.Function, and fully manual gradients.
  
## per_lin_tanh_bce.ipynb  
Implements `Linear → Tanh → BCE` with the same three levels of abstraction. The notebook compares learning performance of _tanh_ vs. _sigmoid_.
  
## ploss.ipynb  
Describes perceptron loss — a piecewise‑linear loss used with a hard sign activation. Historically important as it matches Rosenblatt’s original 1957 learning rule. The notebook implements a custom backward pass.
  
## probability.ipynb  
Demonstrate how to calculate probabilities and conditional probabilities.  
  
## relu.ipynb  
Rectified Linear Unit (ReLU) is a piecewise‑linear activation that preserves positive values and zeroes out negatives. Its non‑saturating gradient was a major breakthrough enabling deep networks to train effectively.
  
## sigmoid.ipynb  
Sigmoid maps real inputs to (0, 1). Historically important but prone to saturation and vanishing gradients. The notebook implements _sigmoid_ with a custom backward pass.
  
## sign.ipynb 
The sign function maps inputs to {−1, 0, +1}. It descends from the McCulloch–Pitts threshold unit and blocks gradient flow, making it incompatible with standard backpropagation. The notebook implements a custom backward function.
  
## tanh.ipynb  
Tanh maps inputs to (−1,+1). Its derivative `1−tanh^2(x)` is twice as large as the _sigmoid_ derivative around zero, giving stronger gradients and typically faster convergence. The notebook implements tanh with a custom backward pass.
