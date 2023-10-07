from finn import Visualiser
from finn import Finn
import torch


def run1d():
	steps = 1000
	dim = 1
	pos = True
	area = 1.
	condition = lambda a: True
	x_lim_lower = -1.
	x_lim_upper = 1.
	f = Finn(
			dim=dim, 
			pos=pos,
			condition=condition, 
			area=area, 
			x_lim_lower=x_lim_lower*torch.ones(dim), 
			x_lim_upper=x_lim_upper*torch.ones(dim)
			)
	x = torch.linspace(x_lim_lower, x_lim_upper, steps).unsqueeze(1).expand(-1, dim)
	y = f(x)
	dx = x[1,0] - x[0,0]
	integral = torch.sum(y) * dx
	print("integral:", integral.detach())
	vis = Visualiser()
	vis.update(f, ninputs=1, visdim=[0], lim=[-1,1], step=1/steps)
	input()


def run2d():
	steps = 1001
	vis_step = 0.1
	dim = 2
	pos = True
	area = 10
	condition = lambda a: True
	x_lim_lower = torch.tensor([-3,-3])
	x_lim_upper = torch.tensor([3,3])
	vis_lower = torch.tensor([-10,-10])
	vis_upper = torch.tensor([10,10])
	f = Finn(
		dim=dim,
		pos=pos,
		condition=condition, 
		area=area, 
		x_lim_lower=x_lim_lower, 
		x_lim_upper=x_lim_upper, 
		)
	x = torch.linspace(x_lim_lower[0], x_lim_upper[0], steps)
	y = torch.linspace(x_lim_lower[1], x_lim_upper[1], steps)
	x, y = torch.meshgrid(x, y)
	v = torch.stack([x.reshape(steps*steps), y.reshape(steps*steps)], dim=1)
	z = f(v)
	dx = x[1,0] - x[0,0]
	dy = y[0,1] - y[0,0]
	integral = torch.sum(z) * (dx * dy)
	print("integral", integral)
	vis = Visualiser()
	vis.update(f, ninputs=2, visdim=[0, 1], lim=[vis_lower[0],vis_upper[0],vis_lower[1],vis_upper[1]], step=vis_step)
	# vis2 = Visualiser2D()
	# vis2.update(f.forward_int, transparent=True, ninputs=2, visdim=[0, 1], lim=[x_lim_lower[0],x_lim_upper[0],x_lim_lower[1],x_lim_upper[1]], step=1/steps)
	input()


if __name__ == "__main__":
	run2d()

