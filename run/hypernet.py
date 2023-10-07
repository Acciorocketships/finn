from finn.mlp import MLP, HyperLinear
from torch import nn
import torch

def run1():
	input_dim = 2
	param_dim = 1
	output_dim = 1
	hyper_net = MLP(layer_sizes=[input_dim,4,output_dim], layer_type=HyperLinear)
	num_params = hyper_net.num_params()
	param_net = MLP(layer_sizes=[param_dim,8,num_params], layer_type=nn.Linear)

	batch = 8
	x = torch.randn(batch,input_dim)
	p = torch.randn(batch, param_dim)

	params = param_net(p)
	hyper_net.update_params(params)
	y = hyper_net(x)

	breakpoint()



if __name__ == "__main__":
	run1()