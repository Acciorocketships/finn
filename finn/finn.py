import torch
from finn.mlp import IntegralNetwork
import time

class Finn(torch.nn.Module):

	def __init__(self, dim, pos=False, x_lim_lower=None, x_lim_upper=None, condition=None, area=1.):
		'''
		:param dim: dimension of the input (output dim is 1)
		:param pos: if true, then the constraint f(x) > 0 is added
		:param x_lim_lower: lower limit of integration for x (default is -inf)
		:param x_lim_upper: upper limit of integration for x (default is inf)
		:param condition: function that takes in area, and returns whether or not to apply transformation (bool)
		                  for example, an equality constraint will have condition = lambda area: True
		                  for ∫f(x) <= eps, it would be condition = lambda area: area > eps
		                  to apply no condition, set condition = None (equivalent but more efficient than lambda area: False)
		:param area: the area to set the region to if the condition is true. for the inequality example above, area = eps
		'''
		super().__init__()
		self.dim = dim
		self.x_lim_lower = torch.as_tensor(x_lim_lower)
		self.x_lim_upper = torch.as_tensor(x_lim_upper)
		self.pos = False
		self.area = area
		self.condition = condition
		self.F = IntegralNetwork(self.dim, 1, pos=pos)
		self.eval_points, self.eval_sign = self.get_eval_points()


	def int(self, x):
		return self.F(x)


	def forward(self, x):
		out = self.differentiate(x)
		if self.condition is not None:
			actual_area = self.calc_area()
			if self.condition(actual_area):
				out *= self.area / actual_area
		return out


	def differentiate(self, x):
		x.requires_grad_(True)
		xi = [x[...,i] for i in range(x.shape[-1])]
		y = self.F(torch.stack(xi, dim=-1))
		last_dy = y
		for i in range(self.dim):
			last_dy = torch.autograd.grad(last_dy.sum(), xi[i], retain_graph=True, create_graph=True)[0]
		return last_dy


	def calc_area(self):
		evals = self.F(self.eval_points)[:,0]
		signed_evals = evals * self.eval_sign
		area = signed_evals.sum(dim=0)
		if self.x_lim_upper is None and self.x_lim_lower is None:
			return 1
		if self.x_lim_upper is None:
			return 1 - area
		return area


	def get_eval_points(self):
		if self.x_lim_upper is None and self.x_lim_lower is None:
			return None, None
		if self.x_lim_lower is None:
			pts = self.x_lim_upper.unsqueeze(0)
			eval_sign = torch.tensor([1])
			return pts, eval_sign
		if self.x_lim_upper is None:
			pts = self.x_lim_lower.unsqueeze(0)
			eval_sign = torch.tensor([-1])
			return pts, eval_sign
		pts = torch.zeros(2**self.dim, self.dim)
		eval_sign = torch.ones(2**self.dim)
		for i in range(self.dim):
			xi_lim = torch.tensor([self.x_lim_lower[i], self.x_lim_upper[i]])
			rep_len = int(2 ** i)
			rep_int_len = int(pts.shape[0] / 2 / rep_len)
			pts[:,i] = xi_lim.repeat_interleave(rep_int_len).repeat(rep_len)
			eval_sign *= torch.tensor([-1, 1]).repeat_interleave(rep_int_len).repeat(rep_len)
		return pts, eval_sign
