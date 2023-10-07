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


class IntegralNetwork(nn.Module):

	def __init__(self, input_dim, output_dim, k=4, pos=False):
		super().__init__()
		self.nets = nn.ModuleList([
				build_mlp(input_dim, 
						  output_dim, 
						  nlayers=4, 
						  midmult=4., 
						  layer_type=LinearAbs if pos else nn.Linear, 
						  activation=IntegralActivation, 
						  last_activation=None, 
						  activation_kwargs={"n":input_dim}
						  )
			for _ in range(k)])
		def ind_forward(x):
			out = [self.nets[i](x) for i in range(k)]
			return sum(out)
		self.nets.forward = ind_forward

	def forward(self, x):
		return self.nets(x)


# class MLP(nn.Module):
# 	def __init__(
# 		self,
# 		layer_sizes,
# 		batchnorm=False,
# 		layernorm=False,
# 		monotonic=False,
# 		activation=nn.Mish,
# 		last_activation=None,
# 		activation_kwargs={},
# 	):
# 		super(MLP, self).__init__()
# 		layers = []
# 		for i in range(len(layer_sizes) - 1):
# 			if monotonic:
# 				layers.append(LinearAbs(layer_sizes[i], layer_sizes[i + 1]))
# 			else:
# 				layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
# 			if i < len(layer_sizes) - 2:
# 				if batchnorm:
# 					layers.append(nn.BatchNorm(layer_sizes[i + 1]))
# 				if layernorm:
# 					layers.append(nn.LayerNorm(layer_sizes[i + 1]))
# 				layers.append(activation(**activation_kwargs))
# 			elif i == len(layer_sizes) - 2:
# 				if last_activation is not None:
# 					layers.append(last_activation(**activation_kwargs))
# 		self.net = nn.Sequential(*layers)

# 	def forward(self, X):
# 		return self.net(X)

class MLP(nn.Module):
	def __init__(
		self,
		layer_sizes,
		batchnorm=False,
		activation=nn.Mish,
		last_activation=None,
		layer_type=nn.Linear,
		activation_kwargs={},
	):
		super(MLP, self).__init__()
		layers = []
		for i in range(len(layer_sizes) - 1):
			layers.append(layer_type(layer_sizes[i], layer_sizes[i + 1]))
			if i < len(layer_sizes) - 2:
				if batchnorm:
					layers.append(BatchNorm(layer_sizes[i + 1]))
				layers.append(activation(**activation_kwargs))
			elif i == len(layer_sizes) - 2:
				if last_activation is not None:
					layers.append(last_activation(**activation_kwargs))
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)

	def num_params(self):
		num_params = 0
		for layer in self.net:
			if isinstance(layer, HyperLinear):
				num_params += layer.num_params()
		return num_params

	def update_params(self, params):
		i = 0
		for layer in self.net:
			if isinstance(layer, HyperLinear):
				layer_num_params = layer.num_params()
				layer_params = params[:,i:i+layer_num_params]
				i += layer_num_params
				layer.update_params(layer_params)


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


class HyperLinear(nn.Module):

	def __init__(self, in_dim, out_dim, pos=True):
		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.pos = pos
		self.w = None
		self.b = None

	def num_params(self):
		return self.in_dim * self.out_dim + self.out_dim

	def update_params(self, params):
		# params: b x (in_dim * out_dim + out_dim)
		assert params.shape[1] == self.in_dim * self.out_dim + self.out_dim
		batch = params.shape[0]
		self.w = params[:,:self.in_dim*self.out_dim].view(batch, self.in_dim, self.out_dim)
		self.b = params[:,self.in_dim*self.out_dim:].view(batch, self.out_dim)
		if self.pos:
			self.w = torch.abs(self.w)

	def forward(self, x):
		# x: b x in_dim OR b x n x in_dim
		w = self.w
		b = self.b
		assert x.shape[0] == w.shape[0]
		assert x.shape[-1] == w.shape[1]
		squeeze_output = False
		if x.dim() == 2:
			squeeze_output = True
			x = x.unsqueeze(1)
		if b.dim() == 3:
			b = b.squeeze(1)
		xw = torch.bmm(x,w)
		out = xw + b[:,None]
		if squeeze_output:
			out = out.squeeze(1)
		return out