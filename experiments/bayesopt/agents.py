import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from fractions import Fraction
from vbll_fifo import Regression, FifoVBLL, FifoLaplaceDiag
from methods import low_rank_filter as lofi
from methods import low_rank_last_layer as ll_lrkf
from methods import low_rank_filter_revised as lrkf
from methods import gaussian_process as gp


class VBLLMLP(nn.Module):
    """
    Surrogate MLP for the variational Bayesian last-layer VBLL
    """
    n_hidden: int = 180
    wishart_scale: float = 0.1
    regularization_weight: float = 1 / 10.0

    @nn.compact
    def encode(self, x):
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        return x

    @nn.compact
    def __call__(self, x):
        x = self.encode(x)
        x = Regression(
            in_features=self.n_hidden, out_features=1,
            wishart_scale=self.wishart_scale,
            regularization_weight=self.regularization_weight,
        )(x)
        return x


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
    

def load_lrkf_agent(
        X, rank, cov=1.0, low_rank_diag=False, obs_noise=0.0, dynamics_cov=0.0
):
    surrogate = MLPSurrogate()

    def cov_fn(y): return obs_noise # Function interpolation does not require observation noise
    agent = lrkf.LowRankCovarianceFilter(
        surrogate.apply, cov_fn,
        rank=rank, dynamics_covariance=dynamics_cov,
    )

    def bel_init_fn(key):
        params_init = surrogate.init(key, X)
        bel_init = agent.init_bel(
            params_init,
            cov=cov,
            low_rank_diag=low_rank_diag,
        )
        return bel_init

    return agent, bel_init_fn


def load_ll_lrkf_agent(
        X, rank, cov_hidden=1e-4, cov_last=1.0, low_rank_diag=False,
        obs_noise=0.0, dynamics_hidden=0.0, dynamics_last=0.0,
):
    surrogate = MLPSurrogate()

    def cov_fn(y): return obs_noise # Function interpolation does not require observation noise
    agent = ll_lrkf.LowRankLastLayer(
        surrogate.apply, cov_fn, rank=rank, dynamics_hidden=dynamics_hidden, dynamics_last=dynamics_last,
    )

    def bel_init_fn(key):
        params_init = surrogate.init(key, X)
        bel_init = agent.init_bel(
            params_init,
            cov_hidden=cov_hidden,
            cov_last=cov_last,
            low_rank_diag=low_rank_diag,
        )
        return bel_init

    return agent, bel_init_fn



def load_lofi_agent(
        X, rank, cov_init=1.0, obs_noise=0.0, dynamics_covariance=0.0,
):
    surrogate = MLPSurrogate()

    def cov_fn(y): return obs_noise
    agent = lofi.LowRankPrecisionFilter(
        surrogate.apply, cov_fn, dynamics_covariance=dynamics_covariance, rank=rank, inflate_diag=False,
    )

    def bel_init_fn(key):
        params_init = surrogate.init(key, X)
        bel_init = agent.init_bel(params_init, cov=cov_init)
        return bel_init

    return agent, bel_init_fn


def load_gp_agent(
        X, lenght_scale, nu, buffer_size, obs_noise=0.0
):
    if isinstance(nu, str):
        nu = float(Fraction(nu))

    dim = X.shape[-1]
    lenght_scale = lenght_scale + jnp.sqrt(dim)
    kernel = gp.matern_kernel(length_scale=lenght_scale, nu=nu)
    agent = gp.GaussianProcessRegression(obs_variance=obs_noise, kernel=kernel)

    def bel_init_fn(key):
        bel_init = agent.init_bel(dim_in=dim, buffer_size=buffer_size)
        return bel_init

    return agent, bel_init_fn


def load_fifo_vbll_agent(
    X, learning_rate, buffer_size, n_inner, wishart_scale, regularization_weight
):
    def lossfn(params, counter, x, y, apply_fn):
        res = apply_fn(params, x)
        return res.train_loss_fn(y, counter)

    dim = X.shape[-1]

    surrogate = VBLLMLP(
        wishart_scale=wishart_scale, regularization_weight=regularization_weight
    )
    agent = FifoVBLL(
        surrogate.apply,
        lossfn,
        tx=optax.adamw(learning_rate),
        buffer_size=buffer_size,
        dim_features=dim,
        dim_output=1,
        n_inner=n_inner,
    )

    def bel_init_fn(key):
        params_init = surrogate.init(key, X)
        bel_init = agent.init_bel(params_init)
        return bel_init
    
    return agent, bel_init_fn


def load_ll_laplace_agent(
        X, learning_rate, buffer_size, n_inner, noise=1.0, obs_noise=0.0):

    def lossfn(params, counter, x, y, apply_fn):
        res = apply_fn(params, x).squeeze()
        # logprobas = distrax.Normal(loc=res.squeeze(), scale=noise).log_prob(y.squeeze())
        logprobas = jax.scipy.stats.norm.logpdf(y.squeeze(), res.squeeze(), noise)
        loss = - (logprobas * counter).sum() / counter.sum()
        return loss.squeeze()

    def cov_fn(y): return obs_noise # Function interpolation does not require observation noise
    surrogate = MLPSurrogate()
    agent = FifoLaplaceDiag(
        surrogate.apply,
        cov_fn,
        lossfn,
        tx=optax.adamw(learning_rate),
        buffer_size=buffer_size,
        dim_features=X.shape[-1],
        dim_output=1,
        n_inner=n_inner,
    )

    def bel_init_fn(key):
        params_init = surrogate.init(key, X)
        bel_init = agent.init_bel(params_init)
        return bel_init
    
    return agent, bel_init_fn


AGENTS = {
    "GP": load_gp_agent,
    "laplace": load_ll_laplace_agent,
    "VBLL-greedy": load_fifo_vbll_agent,
    "VBLL": load_fifo_vbll_agent,
    "FLoRES": load_ll_lrkf_agent,
    "LRKF": load_lrkf_agent,
    "LOFI": load_lofi_agent,
}
