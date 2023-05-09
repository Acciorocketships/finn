import numpy as np

from Visualiser2D import Visualiser2D
from Finn import Finn
import torch

def run1d():
	steps = 1000
	dim = 1
	r_inv = lambda x: x
	r_jac_det = lambda u: 1
	u_lim_lower = -1.
	u_lim_upper = 1.
	area = 1.
	f = Finn(dim=dim, r_inv=r_inv, r_jac_det=r_jac_det, u_lim_lower=u_lim_lower*torch.ones(dim), u_lim_upper=u_lim_upper*torch.ones(dim), area=area)
	x = torch.linspace(u_lim_lower, u_lim_upper, steps).unsqueeze(1).expand(-1, dim)
	y = f(x)
	dx = x[1,0] - x[0,0]
	integral = torch.sum(y) * dx
	print("integral", integral)
	vis = Visualiser2D()
	vis.update(f, ninputs=1, visdim=[0], lim=[-1,1], step=1/steps)
	input()


def run2d():
	steps = 1001
	vis_step = 0.1
	dim = 2
	r_inv = lambda x: x #lambda x: torch.stack([torch.atan2(x[:,1],x[:,0]), x.norm(dim=-1)], dim=-1)
	r_jac_det = lambda u: 1 #lambda u: u[:,1]
	u_lim_lower = torch.tensor([-3,-3])
	u_lim_upper = torch.tensor([3,3])
	vis_lower = torch.tensor([-10,-10])
	vis_upper = torch.tensor([10,10])
	area = 10
	f = Finn(
		dim=dim, 
		r_inv=r_inv, 
		r_jac_det=r_jac_det, 
		u_lim_lower=u_lim_lower, 
		u_lim_upper=u_lim_upper, 
		area=area, 
		)
	x = torch.linspace(u_lim_lower[0], u_lim_upper[0], steps)
	y = torch.linspace(u_lim_lower[1], u_lim_upper[1], steps)
	x, y = torch.meshgrid(x, y)
	v = torch.stack([x.reshape(steps*steps), y.reshape(steps*steps)], dim=1)
	z = f(v)
	dx = x[1,0] - x[0,0]
	dy = y[0,1] - y[0,0]
	integral = torch.sum(z) * (dx * dy)
	print("integral", integral)
	vis = Visualiser2D()
	vis.update(f, ninputs=2, visdim=[0, 1], lim=[vis_lower[0],vis_upper[0],vis_lower[1],vis_upper[1]], step=vis_step)
	# vis2 = Visualiser2D()
	# vis2.update(f.forward_int, transparent=True, ninputs=2, visdim=[0, 1], lim=[u_lim_lower[0],u_lim_upper[0],u_lim_lower[1],u_lim_upper[1]], step=1/steps)
	input()


if __name__ == "__main__":
	run2d()

