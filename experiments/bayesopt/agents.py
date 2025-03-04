import flax.linen as nn
from rebayes_mini.methods import low_rank_last_layer as lrll
from rebayes_mini.methods import low_rank_filter as lofi

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

def load_ll_lrkf_agent(
        key, X, rank, cov_hidden=1e-4, cov_last=1.0, low_rank_diag=False,
        obs_noise=0.0, dynamics_hidden=0.0, dynamics_last=0.0,
):
    surrogate = MLPSurrogate()

    def cov_fn(y): return obs_noise # Function interpolation does not require observation noise
    agent = lrll.LowRankLastLayer(
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
        key, X, rank, cov_init=1.0, obs_noise=0.0, dynamics_covariance=0.0,
):
    surrogate = MLPSurrogate()

    def cov_fn(y): return obs_noise
    agent = lofi.LowRankPrecisionFilter(surrogate.apply, cov_fn, dynamics_covariance=dynamics_covariance, rank=rank)

    def bel_init_fn(key):
        params_init = surrogate.init(key, X)
        bel_init = agent.init_bel(params_init, cov=cov_init)
        return bel_init

    return agent, bel_init_fn