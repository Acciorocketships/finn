import optax
import jax.numpy as jnp
import jax
import haiku as hk
from jax import grad, jit, vmap, random
from jax import random
import time

class IntegralModel(hk.Module):
	def __init__(self):
		super().__init__(name="IntegralModel")
		self.net = hk.nets.MLP(output_sizes=[16,16,8,1], activation=jnp.tanh, name='net')

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

def get_f(params, model):
	return lambda x: model.apply(params, x)

def diff_f(params, model, dims):
	f = get_f(params, model)
	diff = lambda f, i: (lambda y: grad(lambda x: jnp.sum(f(x)), argnums=0)(y)[:,i])
	for dim in dims:
		f = diff(f, dim)
	return f

def run():

	batch = 256
	dim = 7
	model = hk.without_apply_rng(hk.transform(model_forward))
	rng = jax.random.PRNGKey(seed=0)
	params = model.init(rng, jnp.ones(dim))
	opt = optax.adamw(learning_rate=1e-3)
	opt_state = opt.init(params)

	for _ in range(10):
		rng, key = random.split(rng)
		x = random.normal(key, (batch, dim))
		y = jnp.expand_dims(jnp.sum(x ** 2, -1), -1)
		grads = eval_grads(params, model, x, y)
		grad_updates, opt_state = opt.update(grads, state=opt_state, params=params)
		params = optax.apply_updates(params, grad_updates)

	df = jit(diff_f(params, model, [slice(None)]))
	dfn = jit(diff_f(params, model, list(range(dim))))
	rng, key = random.split(rng)
	x = random.normal(key, (batch, dim))

	start_time = time.time()
	dfx = df(x)
	df_time = time.time() - start_time
	print("df:", df_time)

	start_time = time.time()
	dfnx = dfn(x)
	dfn_time = time.time() - start_time
	print("dfn:", dfn_time)



if __name__ == "__main__":
	run()