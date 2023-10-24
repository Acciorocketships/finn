import math
import torch
import torch.nn as nn
from torch import Tensor

from finn.activation import IntegralActivation


def build_mlp(input_dim, output_dim, nlayers=1, midmult=1., **kwargs):
	mlp_layers = layergen(input_dim=input_dim, output_dim=output_dim, nlayers=nlayers, midmult=midmult)
	mlp = MLP(layer_sizes=mlp_layers, **kwargs)
	return mlp


def layergen(input_dim, output_dim, nlayers=1, hidden_dim=None, midmult=1.0):
	if hidden_dim is None:
		midlayersize = midmult * (input_dim + output_dim) // 2
		midlayersize = max(midlayersize, 1)
		nlayers += 2
		layers1 = torch.round(
			torch.logspace(math.log10(input_dim), math.log10(midlayersize), steps=(nlayers) // 2)
		).int()
		layers2 = torch.round(
			torch.logspace(
				math.log10(midlayersize), math.log10(output_dim), steps=(nlayers + 1) // 2
			)
		).int()[1:]
		return torch.cat([layers1, layers2]).tolist()
	else:
		return [input_dim] + ([hidden_dim] * (nlayers-1)) + [output_dim]


class IntegralNetwork(nn.Module):

	def __init__(self, input_dim, output_dim, nlayers=3, k=1, pos=False, device='cpu'):
		super().__init__()
		self.nets = nn.ModuleList([
				build_mlp(input_dim,
						  output_dim,
						  nlayers=nlayers,
						  midmult=4.,
						  device=device,
						  layer_type=LinearAbs if pos else nn.Linear,
						  activation=IntegralActivation if pos else nn.Mish,
						  last_activation=None,
						  activation_kwargs={"n":input_dim}
						  )
			for _ in range(k)])
		self.acts = []
		for i in range(k):
			for layer in self.nets[i].net:
				if isinstance(layer, IntegralActivation):
					self.acts.append(layer)

	def forward(self, x):
		return torch.cat([net(x) for net in self.nets], dim=-1).sum(dim=-1).unsqueeze(-1)

	def set_forward_mode(self, mode):
		for act in self.acts:
			act.forward_mode = mode
			act.clear_backward_vals()


class MLP(nn.Module):
	def __init__(
		self,
		layer_sizes,
		activation=nn.Mish,
		last_activation=None,
		layer_type=nn.Linear,
		activation_kwargs={},
		device='cpu',
	):
		super(MLP, self).__init__()
		layers = []
		for i in range(len(layer_sizes) - 1):
			layers.append(layer_type(layer_sizes[i], layer_sizes[i + 1], device=device))
			if i < len(layer_sizes) - 2:
				if activation == IntegralActivation:
					activation_func = activation(**activation_kwargs)
				else:
					activation_func = activation()
				layers.append(activation_func)
			elif i == len(layer_sizes) - 2:
				if last_activation is not None:
					layers.append(last_activation())
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


class LinearAbs(nn.Linear):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, input: Tensor) -> Tensor:
		return nn.functional.linear(input, torch.abs(self.weight), self.bias)