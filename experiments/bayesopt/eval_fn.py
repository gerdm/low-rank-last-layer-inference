import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from jaxopt import ProjectedGradient

class MLPSurrogate(nn.Module):
    n_hidden: int = 180

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        x = nn.Dense(1, name="last_layer")(x)
        return x


def projection(params, hparams, lbound, ubound):
    return jnp.clip(params, lbound, ubound)


def minimise_project(guess_init, fun, lbound, ubound):
    fun_max = lambda x: -fun(x)
    _projection = partial(projection, lbound=lbound, ubound=ubound)
    opt = ProjectedGradient(
        fun=fun_max,
        projection=_projection, # Enforce boundary constraints
        stepsize=1e-5,
        tol=1e-5,
        maxiter=1000,
    )
    res = opt.run(guess_init)
    return res


def step(state, t, key, agent, objective_fn, dim, lbound, ubound):
    bel, y_best = state
    key_step = jax.random.fold_in(key, t)
    key_step, key_guess = jax.random.split(key_step)

    # params_sample = agent.sample_params(key_step, bel).squeeze()
    sampled_fn = agent.sample_fn(key_step, bel)
    # compute location of next best estimate and actual estimate
    # TODO: generalise guess_init and minimise_project
    guess_init = jax.random.uniform(key_guess, shape=(dim,), minval=lbound, maxval=ubound)
    x_next = minimise_project(guess_init, sampled_fn, lbound, ubound).params 
    y_next = objective_fn(x_next)
    # update belief based on true observations
    bel = agent.update(bel, y_next.squeeze(), x_next)

    y_best = y_next * (y_next > y_best) + y_best * (y_next <= y_best)

    out = {
        "x": x_next.squeeze(),
        "y": y_next.squeeze(),
        "y_best": y_best.squeeze()
    }

    state_next = (bel, y_best)
    return state_next, out


def test_run(key, n_steps, agent, bel_init, objective_fn, dim, lbound, ubound, n_warmup=None):
    n_warmup = dim if n_warmup is None else n_warmup
    key_init_x, key_eval = jax.random.split(key)

    # Warmup agent
    x_warmup = jax.random.uniform(key_init_x, shape=(n_warmup, dim), minval=0, maxval=1)
    y_warmup = jax.vmap(objective_fn)(x_warmup)
    bel_init, _ = agent.scan(bel_init, y_warmup, x_warmup)
    y_next = y_warmup[-1]
    
    # Query n_steps
    steps = jnp.arange(n_steps)
    state_init = (bel_init, y_next)
    _eval = partial(step, key=key_eval, agent=agent, objective_fn=objective_fn, dim=dim, lbound=lbound, ubound=ubound)
    bel_final, hist = jax.lax.scan(_eval, state_init, steps)

    return bel_final, hist["y_best"]


@partial(jax.jit, static_argnames=("agent", "n_steps", "objective_fn", "dim", "lbound", "ubound", "n_warmup"))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None))
def test_runs(key, n_steps, agent, bel_init, objective_fn, dim, lbound, ubound, n_warmup):
    return test_run(key, n_steps, agent, bel_init, objective_fn, dim, lbound, ubound, n_warmup)
