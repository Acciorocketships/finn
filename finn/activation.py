import sympy
import sympytorch
import torch


class IntegralActivation(torch.nn.Module):

	def __init__(self, n):
		super().__init__()
		if n == 1:
			self.f = lambda x: (torch.erf(x) + 1) / 2
		elif n == 2:
			self.f = lambda x: x*torch.erf(x)/2 + x/2 + torch.exp(-x**2)/(2*(torch.pi**0.5))
		else:
			self.f = self.erf_nth_int(n)


	def erf_nth_int(self, n):
		x_ = sympy.Symbol('x')
		erf0 = (sympy.functions.special.error_functions.erf(x_) + 1) / 2
		erfi = erf0
		for _ in range(n-1):
			erfi = sympy.integrate(erfi, x_)
		act = sympytorch.SymPyModule(expressions=[erfi], extra_funcs={sympy.core.numbers.Pi: lambda: torch.pi})
		f = lambda x: act(x=x).squeeze(-1)
		return f


	def forward(self, x):
		return self.f(x)


## Tests ##


def test0():
	from visualiser import Visualiser2D
	vis = Visualiser2D()
	x_ = sympy.Symbol('x')
	erf = (sympy.functions.special.error_functions.erf(x_) + 1) / 2
	erf_int = sympy.integrate(erf, x_)
	act = sympytorch.SymPyModule(expressions=[erf_int], extra_funcs={sympy.core.numbers.Pi: lambda: torch.pi})
	f = lambda x: act(x=x)
	vis.update(f, ninputs=1, lim=[-2,2], step=0.05)
	breakpoint()


def test1():
	from visualiser import Visualiser2D
	vis = Visualiser2D()
	f = IntegralActivation(1)
	vis.update(f, ninputs=1, lim=[-2, 2], step=0.05)
	input()


if __name__ == "__main__":
	test1()