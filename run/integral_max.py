from finn import Visualiser
from finn.finn_max import FinnMax
import torch


def run1d():
	steps = 1000
	dim = 1
	pos = True
	area = 1.
	max = 0.1
	x_lim_lower = -1.
	x_lim_upper = 1.
	f = FinnMax(
			dim=dim, 
			pos=pos,
			area=area, 
			x_lim_lower=x_lim_lower*torch.ones(dim), 
			x_lim_upper=x_lim_upper*torch.ones(dim),
			)
	x = torch.linspace(x_lim_lower, x_lim_upper, steps).unsqueeze(1).expand(-1, dim)
	y = f(x, max=max * torch.ones(x.shape[0]))
	dx = x[1,0] - x[0,0]
	integral = torch.sum(y) * dx
	print("integral:", integral.detach())


def run2d():
	steps = 1001
	dim = 2
	pos = True
	area = 10.
	max = 0.8
	x_lim_lower = torch.tensor([-3,-3])
	x_lim_upper = torch.tensor([3,3])
	f = FinnMax(
		dim=dim,
		pos=pos,
		area=area, 
		x_lim_lower=x_lim_lower, 
		x_lim_upper=x_lim_upper,
		)
	x = torch.linspace(x_lim_lower[0], x_lim_upper[0], steps)
	y = torch.linspace(x_lim_lower[1], x_lim_upper[1], steps)
	x, y = torch.meshgrid(x, y)
	v = torch.stack([x.reshape(steps*steps), y.reshape(steps*steps)], dim=1)
	z = f(v, max=max * torch.ones(v.shape[0]))
	dx = x[1,0] - x[0,0]
	dy = y[0,1] - y[0,0]
	integral = torch.sum(z) * (dx * dy)
	print("integral", integral)


if __name__ == "__main__":
	run2d()

