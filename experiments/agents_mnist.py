import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from vbll_fifo import FifoLaplaceDiag
from rebayes_mini.methods import low_rank_filter as lofi 
from rebayes_mini.methods import low_rank_last_layer as flores
from rebayes_mini.methods import low_rank_filter_revised as lrkf

class CNN(nn.Module):
    num_actions: int = 10

    @nn.compact
    def __call__(self, x):
        x = x if len(x.shape) > 3 else x[None, :]
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.elu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=16, kernel_size=(5, 5), padding="VALID")(x)
        x = nn.elu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=120)(x)
        x = nn.elu(x)
        x = nn.Dense(features=84)(x)
        x = nn.elu(x)
        x = nn.Dense(self.num_actions, name="last_layer")(x)
        return x.squeeze()


def mean_fn(params, x, model):
    x = jnp.atleast_2d(x)
    n_in = len(x[0].ravel())
    if n_in == 28 ** 2 + 1:
        # Passing context and action
        action = x[:, 0].astype(int)
        x = x[:, 1:].reshape(-1, 28, 28, 1)
        eta = model.apply(params, x)
        mean = jax.nn.sigmoid(eta[action])
        return jnp.atleast_1d(mean)
    elif n_in == 28 ** 2:
        # Passing context only
        x = x.reshape(-1, 28, 28, 1)
        eta = model.apply(params, x)
        return eta
    else:
        raise ValueError("Undefined input dimension")


def cov_fn(eta, eps=1e-4):
    mean = jax.nn.softmax(jnp.atleast_1d(eta))
    return jnp.diag(mean) - jnp.outer(mean, mean) + jnp.eye(len(mean)) * eps


def lossfn(params, counter, x, y, apply_fn, noise=0.01):
    x = jnp.atleast_2d(x)
    action = x[:, 0].astype(int)
    x = x[:, 1:].reshape(-1, 28, 28, 1)
    res = apply_fn(params, x)

    # params_flat, _ = ravel_pytree(params)
    
    yhat = jnp.take_along_axis(jnp.atleast_2d(res), action[:, None], axis=-1).squeeze()
    yhat = jax.nn.sigmoid(yhat)
    y = y.squeeze()

    log_likelihood = y * jnp.log(yhat) + (1 - y) * jnp.log1p(-yhat)
    log_likelihood = (log_likelihood * counter).sum() / counter.sum()
    return -log_likelihood #+ 0.1 * jnp.power(params_flat, 2).mean()


### Defining agents ###
model = CNN(num_actions=10)


def agent_ogd_adamw(n_inner=5, learning_rate=1e-4):
    buffer_size = 1
    agent = FifoLaplaceDiag(
        partial(mean_fn, model=model),
        cov_fn,
        lossfn,
        tx=optax.adamw(learning_rate),
        buffer_size=buffer_size,
        dim_features = 28 ** 2 + 1,
        dim_output=1,
        n_inner=n_inner,
    )
    return agent, {}


def agent_ogd_muon():
    buffer_size = 1
    n_inner = 1
    learning_rate = 1e-4
    agent = FifoLaplaceDiag(
        partial(mean_fn, model=model),
        cov_fn,
        lossfn,
        tx=optax.contrib.muon(learning_rate),
        buffer_size=buffer_size,
        dim_features = 28 ** 2 + 1,
        dim_output=1,
        n_inner=n_inner,
    )
    return agent, {}


def agent_flores(cov_init_hidden=0.1, cov_init_last=0.1, dynamics_hidden=1e-6, dynamics_last=1e-6):
    agent = flores.LowRankLastLayer(
        partial(mean_fn, model=model),
        cov_fn,
        rank=50,
        dynamics_hidden=dynamics_hidden,
        dynamics_last=dynamics_last,
    )
    init_params = {
        "low_rank_diag": True,
        "cov_hidden": cov_init_hidden,
        "cov_last": cov_init_last,
    }
    return agent, init_params


def agent_lrkf(cov_init=1.0, dynamics_covariance=1e-6):
    agent = lrkf.LowRankCovarianceFilter(
        partial(mean_fn, model=model),
        cov_fn,
        rank=50,
        dynamics_covariance=dynamics_covariance,
    )

    init_params = {
        "low_rank_diag": True,
        "cov": cov_init,
    }

    return agent, init_params


def agent_lofi(cov_init=1.0, dynamics=1e-4, rank=50):
    agent = lofi.LowRankPrecisionFilter(
        partial(mean_fn, model=model),
        partial(cov_fn, eps=0.1),
        dynamics_covariance=dynamics,
        rank=rank
    )
    init_params = {
        "cov": cov_init
    }
    return agent, init_params



agents = {
    "LRKF": agent_lrkf,
    "FLoRES": agent_flores,
    "adamw": agent_ogd_adamw,
    "muon": agent_ogd_muon,
    "LoFi": agent_lofi
}
