from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import torch


class Visualiser2D:

	def __init__(self, ax=None):
		self.ax = ax
		if ax is None:
			fig = plt.figure()
		plt.ion()
		plt.show()


	def clear(self):
		plt.cla()


	def update(self, func, lim=[-1,1,-1,1], step=0.1, transparent=False, visdim=[0,1], ninputs=2, defaultval=0., label=None, cmap=cm.viridis):

		if ninputs >= 2:
			if self.ax is None:
				self.ax = plt.axes(projection='3d')
			x = torch.arange(lim[0],lim[1],step)
			y = torch.arange(lim[2],lim[3],step)
			x, y = torch.meshgrid(x, y)
			v = torch.ones(x.shape[0]*x.shape[1], ninputs) * defaultval
			v[:,visdim[0]] = x.reshape(x.shape[0]*x.shape[1])
			v[:,visdim[1]] = y.reshape(x.shape[0]*x.shape[1])
			z = func(v)
			z = z.view(x.shape).detach().numpy()

			if transparent:
				self.ax.plot_wireframe(x, y, z, label=label)
			else:
				surf = self.ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, label=label)
				surf._edgecolors2d = surf._edgecolor3d
				surf._facecolors2d = surf._facecolor3d

		elif ninputs == 1:
			if self.ax is None:
				self.ax = plt.axes()
			x = torch.tensor(np.arange(lim[0],lim[1],step))
			x = x.float().view(-1, 1)
			z = func(x)
			x = x.detach().numpy()
			z = z.view(x.shape).detach().numpy()
			if transparent:
				self.ax.plot(x, z, 'b:', label=label)
			else:
				self.ax.plot(x, z, 'r', label=label)

		plt.draw()
		plt.pause(0.01)


if __name__ == "__main__":
	vis = Visualiser2D()
	func = lambda x: x[:,0]**2 + x[:,1]
	vis.update(func)