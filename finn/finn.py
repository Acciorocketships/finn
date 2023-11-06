import torch
from torch.func import grad

from finn.mlp import IntegralNetwork


class Finn(torch.nn.Module):

	def __init__(self, dim, pos=False, x_lim_lower=None, x_lim_upper=None, condition=True, area=1., nlayers=2, device='cpu'):
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
		assert self.x_lim_upper.shape == self.x_lim_lower.shape
		assert self.x_lim_upper.shape == (self.dim,)
		self.area = area
		self.condition = condition
		self.F = IntegralNetwork(self.dim, 1, nlayers=nlayers, pos=pos, device=device)
		self.f = self.build_f()
		self.eval_points, self.eval_sign = self.get_eval_points()


	def build_f(self):
		df = self.F
		for i in range(self.dim):
			df = self.df_dxi(df, i)
		f = lambda x: df(x).unsqueeze(-1)
		return f


	def df_dxi(self, f, i):
		return lambda y: grad(lambda x: f(x).sum())(y)[...,i]


	def int(self, x):
		return self.F(x)


	def forward(self, x):
		# out = self.f(x)
		out = self.differentiate(x)
		if self.condition:
			actual_area = self.calc_area()
			if (self.condition==True) or self.condition(actual_area):
				out *= self.area / actual_area
		return out


	def differentiate(self, x):
		self.F.set_forward_mode(True)
		with torch.enable_grad():
			x.requires_grad_(True)
			xi = [x[...,i] for i in range(x.shape[-1])]
			dyi = self.F(torch.stack(xi, dim=-1))
			for i in range(self.dim):
				# start_time = time.time()
				dyi = torch.autograd.grad(dyi.sum(), xi[i], retain_graph=True, create_graph=True, materialize_grads=True)[0]
				# grad_time = time.time() - start_time
				# print(grad_time)
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
