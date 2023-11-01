import time

import torch
from torch.func import grad

from finn.mlp import IntegralNetwork


class FinnMax(torch.nn.Module):

	def __init__(self, dim, pos=False, x_lim_lower=None, x_lim_upper=None, area=1., nlayers=2, device='cpu'):
		'''
		:param dim: dimension of the input (output dim is 1)
		:param pos: if true, then the constraint f(x) > 0 is added
		:param x_lim_lower: lower limit of integration for x (default is -inf)
		:param x_lim_upper: upper limit of integration for x (default is inf)
		:param condition: function that takes in area, and returns whether or not to apply transformation (bool)
		                  for ∫f(x) <= eps, it would be condition = lambda area: area > eps
		                  set to True for an equality constraint
		                  set to False for no constraints on the integral
		:param area: the area to set the region to if the condition is true. for the inequality example above, area = eps
		'''
		super().__init__()
		self.dim = dim
		self.device = device
		self.x_lim_lower = torch.as_tensor(x_lim_lower,device=self.device) if (x_lim_lower is not None) \
							else -torch.ones(dim, device=self.device)
		self.x_lim_upper = torch.as_tensor(x_lim_upper,device=self.device) if (x_lim_lower is not None) \
							else torch.ones(dim, device=self.device)
		self.area = area
		self.domain = torch.prod(self.x_lim_upper - self.x_lim_lower)
		self.F = IntegralNetwork(self.dim, 1, nlayers=nlayers, pos=pos, device=device)
		self.h = torch.nn.Parameter(torch.tensor(0.))
		self.max_h = (self.area / self.domain) * 0.9
		self.eval_points, self.eval_sign = self.get_eval_points()
		self.h.register_hook(self.h_hook)


	def int(self, x):
		return self.F(x)


	def forward(self, x, max=None):
		f_f = self.differentiate(x)
		actual_area = self.calc_area()
		if max is not None:
			self.calc_h(f_f=f_f, f_max=max, v_f=actual_area)
		self.apply_h_constraints() # redundant, no harm
		s = (self.area - self.domain * self.h) / actual_area
		f_d = f_f * s + self.h
		return f_d


	def calc_h(self, f_f, f_max, v_f):
		# calculate h for samples where output exceeds max
		s = (self.area - self.domain * self.h) / v_f
		mask = (f_f * s > f_max)
		h = ((f_f * self.area) - (f_max * v_f)) / ((f_f * self.domain) - v_f)
		h_mask = torch.cat([h[mask], torch.zeros(1)])
		# take maximum over all samples and existing self.h, applying necessary constraints
		h_max = torch.max(h_mask)
		self.h.data = torch.maximum(self.h, h_max)
		self.apply_h_constraints()


	def differentiate(self, x):
		self.F.set_forward_mode(True)
		with torch.enable_grad():
			x.requires_grad_(True)
			xi = [x[...,i] for i in range(x.shape[-1])]
			dyi = self.F(torch.stack(xi, dim=-1))
			for i in range(self.dim):
				dyi = torch.autograd.grad(dyi.sum(), xi[i], retain_graph=True, create_graph=True, materialize_grads=True)[0]
		self.F.set_forward_mode(False)
		return dyi


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
			eval_sign = torch.tensor([1], device=self.device)
			return pts, eval_sign
		if self.x_lim_upper is None:
			pts = self.x_lim_lower.unsqueeze(0)
			eval_sign = torch.tensor([-1], device=self.device)
			return pts, eval_sign
		pts = torch.zeros(2**self.dim, self.dim, device=self.device)
		eval_sign = torch.ones(2**self.dim, device=self.device)
		for i in range(self.dim):
			xi_lim = torch.tensor([self.x_lim_lower[i], self.x_lim_upper[i]], device=self.device)
			rep_len = int(2 ** i)
			rep_int_len = int(pts.shape[0] / 2 / rep_len)
			pts[:,i] = xi_lim.repeat_interleave(rep_int_len).repeat(rep_len)
			eval_sign *= torch.tensor([-1, 1], device=self.device).repeat_interleave(rep_int_len).repeat(rep_len)
		return pts, eval_sign


	def apply_h_constraints(self):
		self.h.data = torch.clamp(self.h, 0., self.max_h)

	def h_hook(self, grad):
		self.apply_h_constraints()