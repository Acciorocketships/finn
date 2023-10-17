# FINN
[Paper](https://arxiv.org/abs/2307.14439)

<img src="img/learned.png" alt="Learned Fixed Integral Network" width=40%> <img src="img/ground_truth.png" alt="Grouth Truth" width=39.65%>

## Description
FINN provides a framework for computing _analytical_ integrals of learned functions. The framework provides a learnable function $f: \mathbb{R}^n \to \mathbb{R}$, which can be trained with any user-defined loss function. Given this function $f$, it defines the analytical integral $\int f(\vec{x}) d\vec{x}$, which can be evaluated in constant time (_i.e._, not requiring a Riemann sum over the domain), and without any additional training (_i.e._, the integral is implicitly defined, not learned separately).

In addition to this standard functionality, we provide two useful constraints that can optionally be applied:
1. *Integral constraint*. This allows us to apply equality or inequality constraints to the integral of $f$. For example, this allows us to parametrise the class of functions such that $\int f(x) dx \leq \epsilon$
2. *Positivity constraint*. This simply ensures that $f(x) \geq 0$. While relatively simple, this constraint is required in most (but not all) applications of FINN. For example, we must use the positivity constraint when using FINN to represent probability distributions.

While FINN can be used in many applications, a few of the most prominent use cases are:
1. Integrating functions without closed-form solutions
2. Applying integral-based constraints to neural networks
3. Representing arbitrary continuous probability distributions

Let us consider an example. Imagine we wish to learn a vector-valued function $h: $\mathbb{R}^n \to \mathbb{R}^d$ subject to the constraint $\int ||h(\vec{x})|| d\vec{x} \leq \epsilon$ (where $d\vec{x}$ indicates the integral over each dimension). We can split this into two functions $g: \mathbb{R}^n \to \mathbb{R}^d$ and $f: \mathbb{R}^n \to \mathbb{R}$, representing the direction ($g$) and magnitude ($f$) of $h$. To represent $g$, we can use a standard neural network and divide the output by its magnitude to obtain a normalised direction vector. Then, we can use FINN to represent $f$. In this use case of FINN, we apply both the inequality integral constraint and the positivity constraint. We must also provide FINN with the desired area, and the limits of integration between which the function should integrate to that area. This completes our parametrisation of $h$, which is given by: $h(\vec{x}) = f(\vec{x})g(\vec{x})$.

## Installation

```bash
cd finn
pip install -e .
```

## Example

```python
steps = 1000
x_lim_lower = -1.
x_lim_upper = 1.
area = 1.
condition = lambda area: True
f = Finn(
  dim=1,
  condition=condition,
  area=area,
  x_lim_lower=x_lim_lower*torch.ones(dim),
  x_lim_upper=x_lim_upper*torch.ones(dim),
)
x = torch.linspace(x_lim_lower, x_lim_upper, steps).unsqueeze(1)
y = f(x)
dx = x[1,0] - x[0,0]
integral = torch.sum(y) * dx
print("integral:", integral) # numerically validate that the integral is 1.0
```
>```
>integral: tensor(1.0008)
>```
