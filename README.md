# FINN
A Fixed Integral Neural Network
[Paper](https://arxiv.org/abs/2307.14439)

<img src="img/learned.png" alt="Learned Fixed Integral Network" width=40%> <img src="img/ground_truth.png" alt="Grouth Truth" width=39.65%>

## Description
FINN is a learnable neural network of any input dimension $\mathbb{R}^n \to \mathbb{R}$ which is constrained to integrate to a given value over a given interval. This constraint is applied analytically, so the integral is exact (not computed numerically) and guaranteed (it is not enforced by an optimisation objective). By default, the value of the function is constrained to be positive, but this constraint can be disabled if necessary. 

While FINN can be used in many applications, a few of the most prominent use cases are:
1. Applying integral-based constraints to neural networks
2. Integrating learned functions
3. Representing arbitrary continuous probability distributions

For example, consider a vector-valued network $v = f(x)$. If we wish to impose the constraint $\int ||f(x)||dx = \epsilon$, then we can use FINN to separately learn the magnitude $m = g(x)$. Then, we can scale the vector v by the learned magnitude: $v' = \frac{m}{||v||}$. This guarantees that the integral of the magnitude of $v'$ over the entire domain equals $\epsilon$. FINN can also be used to impose an inequality constraint, if instead we wish to enforce $\int ||f(x)||dx \leq \epsilon$.

## Installation

```bash
cd finn
pip install -e .
```

## Example

```python
area = 10
dim = 2
u_lim_lower = torch.tensor([-2, -2])
u_lim_upper = torch.tensor([2, 2])
f = Finn(
  dim=dim, 
  u_lim_lower=u_lim_lower, 
  u_lim_upper=u_lim_upper, 
  area=area, 
  )
steps = 1001
x = torch.linspace(u_lim_lower[0], u_lim_upper[0], steps)
y = torch.linspace(u_lim_lower[1], u_lim_upper[1], steps)
x, y = torch.meshgrid(x, y)
v = torch.stack([x.reshape(steps*steps), y.reshape(steps*steps)], dim=1)
z = f(v)
dx = x[1,0] - x[0,0]
dy = y[0,1] - y[0,0]
integral = torch.sum(z) * (dx * dy)
print("integral:", integral) # should equal the user-specified area
```
