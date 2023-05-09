import torch
from mlp import MonotonicNetwork

class Finn(torch.nn.Module):

	def __init__(self, dim, r_inv=lambda x: x, r_jac_det=lambda u: 1, u_lim_lower=None, u_lim_upper=None, area=1.):
		'''
		:param dim: dimension of the input (output dim is 1)
		:param r_inv: defines the limits of integration with u substitution u = r_inv(x)
					  for example, if r(u) = <u0 * cos(u1), u0 * sin(u1)>, then r_inv(x) = <tan(x1/x0), sqrt(x0^2 + x1^2)>
		:param r_jac_det: defined as the determinant of the jacobian of r: |∇r(u)|
			 		  for example, if r(u) = <u0 * cos(u1), u0 * sin(u1)>, then |∇r(u)| = u0
		:param u_lim_lower: lower limit of integration for u (default is -inf)
		:param u_lim_upper: upper limit of integration for u (default is inf)
		:param area: the desired value of the integral over the region
		:param allow_negative: determines whether negative values of f(x) are allowed
		'''
		super().__init__()
		self.dim = dim
		self.r_inv = r_inv
		self.r_jac_det = r_jac_det
		self.u_lim_lower = u_lim_lower
		self.u_lim_upper = u_lim_upper
		self.area = area
		self.Fr = MonotonicNetwork(self.dim, 1)
		self.eval_points, self.eval_sign = self.get_eval_points()


	def forward(self, x):
		actual_area = self.calc_area()
		u = self.r_inv(x)
		f = self.differentiate(u)
		scaling = self.area / actual_area / self.r_jac_det(u)
		out = f * scaling
		return out


	def differentiate(self, u):
		u.requires_grad_(True)
		y = self.Fr(u)
		last_dy = y
		for i in range(self.dim):
			dy_all = torch.autograd.grad(last_dy.sum(), u, retain_graph=True, create_graph=True)[0]
			dy = select_index(dy_all, i, -1)
			last_dy = dy
		return last_dy


	def forward_int(self, u):
		return self.Fr(u)[:,0]


	def calc_area(self):
		evals = self.forward_int(self.eval_points)
		signed_evals = evals * self.eval_sign
		area = signed_evals.sum(dim=0)
		if self.u_lim_upper is None and self.u_lim_lower is None:
			return 1
		if self.u_lim_upper is None:
			return 1 - area
		return area


	def get_eval_points(self):
		if self.u_lim_upper is None and self.u_lim_lower is None:
			return None, None
		if self.u_lim_lower is None:
			pts = self.u_lim_upper.unsqueeze(0)
			eval_sign = torch.tensor([1])
			return pts, eval_sign
		if self.u_lim_upper is None:
			pts = self.u_lim_lower.unsqueeze(0)
			eval_sign = torch.tensor([-1])
			return pts, eval_sign
		pts = torch.zeros(2**self.dim, self.dim)
		eval_sign = torch.ones(2**self.dim)
		for i in range(self.dim):
			xi_lim = torch.tensor([self.u_lim_lower[i], self.u_lim_upper[i]])
			rep_len = int(2 ** i)
			rep_int_len = int(pts.shape[0] / 2 / rep_len)
			pts[:,i] = xi_lim.repeat_interleave(rep_int_len).repeat(rep_len)
			eval_sign *= torch.tensor([-1, 1]).repeat_interleave(rep_int_len).repeat(rep_len)
		return pts, eval_sign



def select_index(src, idx, dim):
	idx_list = [slice(None)] * len(src.shape)
	idx_list[dim] = idx
	return src.__getitem__(idx_list)
