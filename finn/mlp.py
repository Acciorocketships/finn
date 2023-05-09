import torch.nn as nn
import torch
from torch import Tensor
import math
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


class MonotonicNetwork(nn.Module):

	def __init__(self, input_dim, output_dim, k=4):
		super().__init__()
		self.nets = nn.ModuleList([
				build_mlp(input_dim, output_dim, nlayers=4, midmult=4., monotonic=True, activation=IntegralActivation, last_activation=None, activation_kwargs={"n":input_dim})
			for _ in range(k)])
		def ind_forward(x):
			out = [self.nets[i](x) for i in range(k)]
			return sum(out)
		self.nets.forward = ind_forward

	def forward(self, x):
		return self.nets(x)


class MLP(nn.Module):
	def __init__(
		self,
		layer_sizes,
		batchnorm=False,
		layernorm=False,
		monotonic=False,
		activation=nn.Mish,
		last_activation=None,
		activation_kwargs={},
	):
		super(MLP, self).__init__()
		layers = []
		for i in range(len(layer_sizes) - 1):
			if monotonic:
				layers.append(LinearAbs(layer_sizes[i], layer_sizes[i + 1]))
			else:
				layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			if i < len(layer_sizes) - 2:
				if batchnorm:
					layers.append(nn.BatchNorm(layer_sizes[i + 1]))
				if layernorm:
					layers.append(nn.LayerNorm(layer_sizes[i + 1]))
				layers.append(activation(**activation_kwargs))
			elif i == len(layer_sizes) - 2:
				if last_activation is not None:
					layers.append(last_activation(**activation_kwargs))
		self.net = nn.Sequential(*layers)

	def forward(self, X):
		return self.net(X)


class LinearAbs(nn.Linear):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, input: Tensor) -> Tensor:
		return nn.functional.linear(input, torch.abs(self.weight), self.bias)


class SigmoidZero(nn.Module):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def forward(self, input: Tensor) -> Tensor:
		return 1 / (1 + torch.pow(input, -2))