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
        stepsize=1e-3,
        tol=1e-6,
        maxiter=500,
    )
    res = opt.run(guess_init)
    return res


def query_next_point_grad(key, fn, dim, lbound, ubound):
    """
    Obtain next point to query my maximising the surrogate function.
    This choice assumes that fn is diferentiable w.r.t. its input

    fn: sampled function to find argmax of
    """
    guess_init = jax.random.uniform(key, shape=(dim,), minval=lbound, maxval=ubound)
    x_next = minimise_project(guess_init, fn, lbound, ubound).params 
    return x_next


def query_next_point_grid(key, fn, dim, lbound, ubound, n_samples=1_000):
    """
    Obtain next point to query by maximising the surrogate function.
    This choice samples from a grid of points and takes the argmax.
    """
    x_eval = jax.random.uniform(key, minval=lbound, maxval=ubound, shape=(n_samples, dim))
    y_surrogate = fn(x_eval)
    x_next = x_eval[y_surrogate.argmax()]
    return x_next


def step(
        state, t, key, agent, objective_fn, dim, lbound, ubound, query_method="grid"
):
    bel, y_best = state
    key_step = jax.random.fold_in(key, t)
    key_step, key_guess = jax.random.split(key_step)

    # params_sample = agent.sample_params(key_step, bel).squeeze()
    sampled_fn = agent.sample_fn(key_step, bel)

    # compute location of next best estimate and actual estimate
    if query_method == "grid":
        x_next = query_next_point_grid(key_guess, sampled_fn, dim, lbound, ubound)
    elif query_method == "grad":
        x_next = query_next_point_grad(key_guess, sampled_fn, dim, lbound, ubound)
    else:
        raise ValueError(f"Query method {query_method} not defined")

    # Obtain true objective
    y_next = objective_fn(x_next)
    # update belief based on true observations
    bel = agent.update(bel, y_next.squeeze(), x_next)

    y_best = y_next * (y_next > y_best) + y_best * (y_next <= y_best)

    out = {
        "x": x_next.squeeze(),
        "y": y_next.squeeze(),
        "y_best": y_best.squeeze(),
        # "rho": bel.rho.squeeze(),
        # "log_weight": bel.log_weight.squeeze(),
        # "ess": 1 / jnp.sum(jnp.exp(bel.log_weight) ** 2),
    }

    state_next = (bel, y_best)
    return state_next, out


def test_run(
        key, n_steps, agent, bel_init_fn, objective_fn,
        dim, lbound, ubound, n_warmup=None, query_method="grid"
):
    n_warmup = dim if n_warmup is None else n_warmup
    key_init_x, key_eval, key_bel = jax.random.split(key, 3)

    bel_init = bel_init_fn(key_bel)

    # Warmup agent
    x_warmup = jax.random.uniform(key_init_x, shape=(n_warmup, dim), minval=lbound, maxval=ubound)
    y_warmup = jax.vmap(objective_fn)(x_warmup)
    bel_init, _ = agent.scan(bel_init, y_warmup, x_warmup)
    y_next = jnp.max(y_warmup, axis=0)
    
    # Query n_steps
    steps = jnp.arange(n_steps)
    state_init = (bel_init, y_next)
    _eval = partial(
        step,
        key=key_eval, agent=agent, objective_fn=objective_fn, dim=dim, lbound=lbound, ubound=ubound,
        query_method=query_method
    )
    (bel_final, _), hist = jax.lax.scan(_eval, state_init, steps)

    return bel_final, hist


# @partial(jax.jit, static_argnames=("agent", "n_steps", "objective_fn", "dim", "lbound", "ubound", "n_warmup"))
@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, None))
def test_runs(key, n_steps, agent, bel_init, objective_fn, dim, lbound, ubound, n_warmup, query_method):
    _, hist = test_run(key, n_steps, agent, bel_init, objective_fn, dim, lbound, ubound, n_warmup, query_method)
    return hist

