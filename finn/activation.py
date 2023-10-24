import os
from pathlib import Path

import dill
import sympy
import sympytorch
import torch
from torch.func import grad


class IntegralActivation(torch.nn.Module):
	def __init__(self, n):
		super().__init__()
		self.n = n
		recompute = True
		filename = str(Path(os.path.dirname(os.path.realpath(__file__))) / Path("act.pkl"))
		if os.path.isfile(filename):
			with open(filename, 'rb') as f:
				self.acts = dill.load(f)
				if n in self.acts:
					recompute = False
		if recompute:
			self.acts = self.compute_modules()
		with open(filename, 'wb') as f:
			dill.dump(self.acts, f)
		self.act = self.create_activation(self.n)
		self.forward_mode = False
		self.register_backward_hook(self.backward_hook)

	def compute_modules(self):
		def squeeze_output(func):
			return lambda x: func(x=x).squeeze(-1)
		acts = {
			0: lambda x: torch.exp(-x**2) / torch.sqrt(torch.tensor(torch.pi)),
			1: lambda x: (torch.erf(x) + 1) / 2,
		}
		x_ = sympy.Symbol('x')
		erfi = (sympy.functions.special.error_functions.erf(x_) + 1) / 2
		for i in range(2, self.n+1):
			erfi = sympy.integrate(erfi, x_)
			erfi_simp = erfi.simplify()
			acti = sympytorch.SymPyModule(expressions=[erfi_simp], extra_funcs={sympy.core.numbers.Pi: lambda: torch.pi})
			acts[i] = squeeze_output(acti)
		return acts

	# def create_activation(self, i):
	# 	if i > 0:
	# 		mod = self.acts[i]
	# 		deriv_mod = lambda x: (self.create_activation(i - 1).apply(x) if (i > 1) else self.acts[0](x))
	#
	# 		class IntAct(torch.autograd.Function):
	# 			@staticmethod
	# 			def forward(x):
	# 				return mod(x)
	# 			@staticmethod
	# 			def setup_context(ctx, inputs, outputs):
	# 				x, = inputs
	# 				ctx.save_for_backward(x)
	# 			@staticmethod
	# 			def backward(ctx, grad_output):
	# 				x, = ctx.saved_tensors
	# 				if i in self.backward_vals:
	# 					dx = self.backward_vals[i]
	# 					real_dx = deriv_mod(x)
	# 					try:
	# 						assert (dx-real_dx).norm() == 0
	# 					except:
	# 						breakpoint()
	# 				else:
	# 					dx = deriv_mod(x)
	# 					self.backward_vals[i] = dx
	# 				return grad_output * dx
	# 		return IntAct
	# 	else:
	# 		return self.acts[0]

	def create_activation(self, i):
		if i > 0:
			mod = self.acts[i]
			deriv_mod = lambda x: (self.create_activation(i - 1).apply(x) if (i > 1) else self.acts[0](x))

			class IntAct(torch.autograd.Function):
				backward_vals = {}
				@staticmethod
				def forward(x):
					return mod(x)
				@staticmethod
				def setup_context(ctx, inputs, outputs):
					x, = inputs
					ctx.save_for_backward(x)
				@staticmethod
				def backward(ctx, grad_output):
					x, = ctx.saved_tensors
					if i in IntAct.backward_vals:
						dx = IntAct.backward_vals[i]
						# real_dx = deriv_mod(x)
						# assert dx.shape == real_dx.shape
						# assert (dx-real_dx).norm() == 0
					else:
						dx = deriv_mod(x)
						IntAct.backward_vals[i] = dx
					return grad_output * dx
			return IntAct
		else:
			return self.acts[0]

	def forward(self, x):
		return self.act.apply(x)

	def clear_backward_vals(self):
		self.act.backward_vals.clear()

	def backward_hook(self, module, grad_input, grad_output):
		if not self.forward_mode:
			self.clear_backward_vals()

## Tests ##

def dfn(f, n):
	df = f
	for i in range(n):
		df = df_dxi(df)
	return lambda x: df(x).unsqueeze(-1)

def df_dxi(f):
	return grad(lambda x: f(x).sum())

def test0():
	from matplotlib import pyplot as plt
	import time
	n = 2
	x = torch.linspace(-1, 1, 100)
	act = IntegralActivation(n)
	# y = act(x)
	df5 = dfn(act, n)
	start_time = time.time()
	df5x = df5(x)
	dt = time.time() - start_time
	plt.plot(x, df5x)
	plt.show()

def test2():
	import time
	model = IntegralActivation(8)
	dmodel = dfn(model, 8)
	x = torch.linspace(-2, 2, 10)
	start_time = time.time()
	dx = dmodel(x)
	dt = time.time() - start_time


if __name__ == "__main__":
	test2()