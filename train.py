import numpy as np
from Visualiser2D import Visualiser2D
from Finn import Finn
import torch

def run():

	# Init
	dim = 2
	u_lim_lower = torch.tensor([-3,-3])
	u_lim_upper = torch.tensor([3,3])
	vis_lower = u_lim_lower
	vis_upper = u_lim_upper
	vis_step = 0.1
	area = 1.
	f = Finn(
		dim=dim, 
		u_lim_lower=u_lim_lower, 
		u_lim_upper=u_lim_upper, 
		area=area
		)

	# Ground Truth
	sigma_x = 1.
	sigma_y = 1.
	mu_x = 0.
	mu_y = 0.5
	g = lambda x: (1 / (2 * torch.pi * sigma_x * sigma_y)) * torch.exp(
		-0.5 * (
			((x[...,0] - mu_x) / sigma_x) ** 2 + 
			((x[...,1] - mu_y) / sigma_y) ** 2
		))

	# Training Init
	batch = 64
	optim = torch.optim.Adam(f.parameters(), lr=1e-3)
	loss_fn = torch.nn.MSELoss()
	vis = Visualiser2D()

	# Training
	for it in range(10000):
		optim.zero_grad()
		x = 6*(torch.rand(batch, dim)-0.5)

		yhat = f(x)
		y = g(x)
		
		loss = loss_fn(yhat, y)
		# aa = torch.autograd.grad(loss, x, retain_graph=True, create_graph=True)[0]
		loss.backward()
		optim.step()

		print(loss)
		if it % 50 == 0:
			vis.clear()
			vis.update(f, ninputs=2, visdim=[0, 1], lim=[vis_lower[0],vis_upper[0],vis_lower[1],vis_upper[1]], step=vis_step)
			vis.update(g, ninputs=2, visdim=[0, 1], lim=[vis_lower[0],vis_upper[0],vis_lower[1],vis_upper[1]], step=vis_step, transparent=True)



if __name__ == "__main__":
	run()