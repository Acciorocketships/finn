from finn import Visualiser
from finn import Finn
import torch

def run():

	# Init
	dim = 2
	x_lim_lower = -5 * torch.ones(2)
	x_lim_upper = 5 * torch.ones(2)
	vis_lower = x_lim_lower
	vis_upper = x_lim_upper
	vis_step = 0.1
	area = 3.
	f = Finn(
		dim=dim, 
		pos=True,
		area=area,
		condition=lambda a: a > area,
		x_lim_lower=x_lim_lower, 
		x_lim_upper=x_lim_upper, 
		)

	# Ground Truth
	def create_g(c, sigma_x, sigma_y, mu_x, mu_y):
		g = lambda x: (c / (2 * torch.pi * sigma_x * sigma_y)) * torch.exp(
			-0.5 * (
				((x[...,0] - mu_x) / sigma_x) ** 2 +
				((x[...,1] - mu_y) / sigma_y) ** 2
			))
		return g
	g0 = create_g(1, 1., 1., 1., 1.)
	g1 = create_g(1, 1., 1., -2., 0.)
	g1 = create_g(0.5, 0.5, 1., -1., 1.)
	g = lambda x: g0(x) + g1(x)

	# Training Init
	batch = 64
	optim = torch.optim.Adam(f.parameters(), lr=1e-3)
	loss_fn = torch.nn.MSELoss()
	vis = Visualiser()

	# Training
	for it in range(10000):
		optim.zero_grad()
		x = 6*(torch.rand(batch, dim)-0.5)

		yhat = f(x).squeeze(-1)
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