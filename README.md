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
The `backward` methods are the core mechanism behind PyTorch autograd system. A custom backward pass receives the gradient from above and must return gradients with respect to each input.
  
```engine=python
class Mul2Function(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return 2 * x

    @staticmethod
    def backward(ctx, grad_output):
        dF_df = grad_output   # upstream gradient
        df_dx = 2             # local derivative
        return (dF_df * df_dx,)
```
  
This notebook implements custom backward functions for a sequence of increasingly complex operations, illustrating how gradient flow is constructed from local derivatives.
  
## bce.ipynb  
Binary cross‑entropy is a loss function used for binary classification. It measures the divergence between a predicted probability 𝑝 and a true binary label 𝑡 ∈ {0, 1}, penalizing confident but incorrect predictions. The notebook derives BCE from first principles and implements it with a custom autograd backward function.
    
## common.ipynb  
Utility functions and helpers shared across notebooks.  
  
## differential.ipynb  
This dojo builds a clear, incremental pathway through differentials, gradients, and Jacobians for a wide variety of function types. Each case includes the differential formula, the appropriate derivative object, and a numerical example with finite‑difference verification. The goal is to show how the same idea — the differential as a linear approximation — manifests across scalar, vector, and matrix domains.  

1. $\mathbb{R} \to \mathbb{R}$ — scalar derivative.  
We begin with the simplest setting: a real function of one real variable.
The differential reduces to the familiar form $df = f'(x)\,dx$. This serves as the conceptual foundation for all later generalizations.

2. $\mathbb{R}^n \to \mathbb{R}$ — gradient.  
For multivariate scalar functions, the derivative becomes the gradient vector.
The differential is the inner product between the gradient and the perturbation: $df = (\nabla f(\mathbf{x}))^\top d\mathbf{x}$. This introduces the idea that the gradient is the unique vector representing the linear approximation.

3. $\mathbb{R} \to \mathbb{R}^n$ — simple Jacobian.  
A scalar input and vector output produce a column‑vector Jacobian.
This is the first example where the derivative is no longer a scalar but a linear map between spaces of different dimensions.

4. $\mathbb{R}^n \to \mathbb{R}^n$ — full Jacobian matrix.  
For vector‑valued functions of vector inputs, the derivative becomes an $n \times n$ Jacobian matrix. Each row corresponds to the gradient of one output component. This case generalizes the idea of directional sensitivity across multiple outputs.

5. $\mathbb{R}^n \to \mathbb{R}^n$ (element‑wise) — diagonal Jacobian. .
For element‑wise functions, the Jacobian becomes diagonal. Each output depends only on its corresponding input coordinate, so cross‑derivatives vanish. This case highlights how structure in the function produces structure in the Jacobian.

6. $\mathbb{R} \times \mathbb{R} \to \mathbb{R}$ — two partial derivatives.  
For functions of two scalar variables, the differential splits into two components: $df = f_{x_1}\,dx_1 + f_{x_2}\,dx_2$. This is the first example of a differential with multiple independent perturbations.

7. $\mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ — two gradients.  
For scalar functions of two vector arguments, the differential becomes the sum of two inner products: $df = (\nabla_{\mathbf{x}} f)^\top d\mathbf{x} + (\nabla_{\mathbf{y}} f)^\top d\mathbf{y}$.  This generalizes the idea of partial derivatives to vector‑valued inputs.

8. $\mathbb{R}^{n \times m} \times \mathbb{R}^{n \times m} \to \mathbb{R}$ — matrix derivatives.  
For matrix‑valued inputs, the derivative is expressed using Frobenius inner products. This case shows how matrix calculus fits naturally into the same differential framework.


## entropy.ipynb  
Entropy quantifies uncertainty: how many bits do we need on average to describe a random variable. The notebook builds intuition for the classic formula:

$$ H(X) = \sum_{i} \Big\{ p_i \Big\}_{\text{how frequently}} \Big\{ \log_2 \frac{1}{p_i} \Big\}_{\text{how many bits}} $$
  
It also demonstrates entropy examples for geometric, uniform, and Zipf‑like distributions, illustrating how uncertainty changes with the shape of the probability mass.
   
## differential.ipynb  
The notebook introduces the concept of differentials as linear approximations to changes in a function’s output based on changes in its input.
    
For any function $f:X \to Y$, between normed vector spaces, the Fréchet derivative $Df(x)$ is the unique linear map satisfying:

$$ f(x + h) = f(x) + Df(x) h + \epsilon \implies$$
  
$$ df = Df(x) dx $$

which specializes to:

$$ df =
\begin{dcases}
f'(x)dx \quad &\text{for } f:\mathbb{R}\to\mathbb{R} \\
\\
\nabla f(x) \cdot dx \quad &\text{for }f:\mathbb{R}^n\to\mathbb{R} \\
\\
J_f(x) \cdot dx \quad &\text{for }f:\mathbb{R}^n\to\mathbb{R}^m
\end{dcases}
$$
  
In practical machine learning, backpropagation operates exactly on these Fréchet differentials, expressed in coordinates as gradients and Jacobians.
  
## inner_product.ipynb  
Introduces the inner product for vectors and generalizes it to matrices via the Frobenius inner product.

$$ \langle A, B \rangle = 
\begin{dcases}
\sum_{i} A_i B_i \quad &\text{for } A, B \in \mathbb{R}^n \\
\\
\sum_{i,j} A_{ij} B_{ij} = \text{tr}(AB^T) \quad &\text{for } A, B \in \mathbb{R}^{m \times n}
\end{dcases}
$$

The notebook also explains why transposition appears naturally in gradient formulas: the inner‑product structure forces gradients to align with the adjoint (transpose) of the corresponding linear map.
  
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
Describes the McCulloch–Pitts neuron, the first mathematical model of a neuron (1943). It consists of a linear combination of inputs ollowed by a hard step sign activation. The model was designed purely for computation, with no learning capability. By manually choosing weights and biases, it can implement Boolean functions such as `and` and `or`, but it cannot represent `xor` due to linear inseparability.
  
## per_lin_mse.ipynb  
Implements a linear perceptron trained with mean squared error — historically known as ADALINE (ADAptive LInear NEuron), introduced by Widrow and Hoff in 1960.
ADALINE was a major milestone in the development of artificial intelligence: the first model to use a continuous output and to update its weights using the true gradient of a differentiable loss function, decades before backpropagation was formalized.
    
## per_lin_ploss.ipynb  
Implements Rosenblatt’s perceptron — the first trainable artificial neuron from 1957. Learning is performed via the perceptron update rule, equivalent to a subgradient step on the perceptron `loss function`.
  
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
Sigmoid maps real inputs (_logits_) to $(0.0, 1.0)$ (_probabilities_) so is used for binary classification. Historically important but prone to saturation and vanishing gradients. The notebook implements sigmoid with a custom backward pass.
  
## sign.ipynb 
The sign function maps inputs (logits) to the discrete set $\{−1, 0, +1\}$, making it a simple thresholding mechanism.
It descends from the McCulloch–Pitts threshold unit, and as a completely non-linear function, it completely blocks gradient flow, which makes it incompatible with standard backpropagation.
The notebook implements the sign function together with a custom backward pass.
  
## tanh.ipynb  
Tanh maps inputs to $(−1.0, \, +1.0)$. Its derivative $1−\tanh^2(x)$ is twice as large as the _sigmoid_ derivative around zero, giving stronger gradients and typically faster convergence. The notebook implements tanh with a custom backward pass.
