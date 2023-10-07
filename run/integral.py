from finn import Visualiser
from finn import Finn
import torch

def run1():
	f = Finn(
			dim=1, 
			area=1.,
			pos=False,
			condition=None,
			x_lim_lower=torch.tensor([-1]), 
			x_lim_upper=torch.tensor([1]),
		)
	vis = Visualiser()
	steps = 100
	vis.update(f, ninputs=1, lim=[-1,1], step=1/steps)
	vis.update(lambda x: f.int(x) - f.int(torch.tensor([-1.])), ninputs=1, lim=[-1,1], step=1/steps, transparent=True)
	input()

def run2():
	f = Finn(
			dim=2, 
			area=1.,
			condition=None,
			x_lim_lower=torch.tensor([-1,-1]), 
			x_lim_upper=torch.tensor([1,1]), 
		)
	vis = Visualiser()
	steps = 100
	vis.update(f, ninputs=2, lim=[-1,1,-1,1], step=1/steps)
	vis.update(lambda x: (f.int(x) - f.int(torch.tensor([-1., -1.]))), ninputs=2, lim=[-1,1,-1,1], step=1/steps, transparent=True)
	input()


if __name__ == "__main__":
	run1()