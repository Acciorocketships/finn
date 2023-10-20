import optax
import jax.numpy as jnp
import jax
import haiku as hk
from jax import grad, jit, vmap, random
from jax import random
import numpy as np
from finn.visualiser import Visualiser

class IntegralModel(hk.Module):
	def __init__(self):
		super().__init__(name="IntegralModel")
		self.net = hk.nets.MLP(output_sizes=[16,16,8,1], activation=mish, name='net')

	def __call__(self, x):
		return self.net(x)

def model_forward(x):
	model = IntegralModel()
	return model(x)

def mish(x):
	return x * jnp.tanh(jax.nn.softplus(x))

def eval_loss(params, model, x, target):
	loss_fn = lambda pred, target: jnp.mean(optax.l2_loss(predictions=pred, targets=target))
	pred = model.apply(params, x)
	loss = loss_fn(pred, target)
	return loss

def eval_grads(params, model, x, target):
	return grad(eval_loss, argnums=0)(params, model, x, target)

def run():

	batch = 256
	dim = 1
	model = hk.without_apply_rng(hk.transform(jit(model_forward)))
	rng = jax.random.PRNGKey(seed=0)
	params = model.init(rng, jnp.ones(dim))
	opt = optax.adamw(learning_rate=1e-3)
	opt_state = opt.init(params)
	vis = Visualiser()

	for _ in range(1000):
		rng, key = random.split(rng)
		x = random.normal(key, (batch, dim))
		y = jnp.expand_dims(jnp.sum(x ** 2, -1), -1)
		loss = eval_loss(params, model, x, y)
		grads = eval_grads(params, model, x, y)
		grad_updates, opt_state = opt.update(grads, state=opt_state, params=params)
		params = optax.apply_updates(params, grad_updates)
		vis.clear()
		vis.update(func=lambda x: np.array(model.apply(params, np.array(x))), ninputs=1, lim=[-1, 1, -0.1, 1])
		vis.update(func=lambda x: x**2, ninputs=1, transparent=True, lim=[-1, 1, -0.1, 1])
		print(loss)



if __name__ == "__main__":
	run()