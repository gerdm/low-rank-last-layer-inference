"""
Source: https://github.com/VectorInstitute/vbll/blob/main/vbll/jax/layers/regression.py

replay-SGD-compatible VBLL
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from collections.abc import Callable
from rebayes_mini.methods.replay_sgd import FifoSGD
from vbll.jax.utils.distributions import Normal, DenseNormal

def KL(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(axis=(-1, -2)) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(axis=-1)
    logdet_term = (feat_dim * jnp.log(q_scale) - p.logdet_covariance).sum(axis=-1)

    return 0.5 * (mse_term + trace_term + logdet_term)

@dataclass
class VBLLReturn:
    predictive: Normal | DenseNormal
    train_loss_fn: Callable[[jnp.array], jnp.array]
    val_loss_fn: Callable[[jnp.array], jnp.array]
    ood_scores: None | Callable[[jnp.array], jnp.array] = None

class Regression(nn.Module):
    in_features: int
    out_features: int
    regularization_weight: float
    parameterization: str = "dense"
    prior_scale: float = 1.0
    wishart_scale: float = 1e-2
    dof: float = 1.0

    @nn.module.compact
    def __call__(self, x, y=None):
        adjusted_dof = (self.dof + self.out_features + 1.) / 2.
        noise_mean = self.param("noise_mean", nn.initializers.zeros, (self.out_features,))
        noise_logdiag = self.param("noise_logdiag", nn.initializers.normal(), (self.out_features,))
        W_mean = self.param("W_mean", nn.initializers.normal(), (self.out_features, self.in_features))
        W_logdiag = self.param("W_logdiag", nn.initializers.normal(), (self.out_features, self.in_features))
        if self.parameterization == "dense":
            W_offdiag = self.param("W_offdiag", lambda rng, shape: jax.random.normal(rng, shape) / self.in_features, (self.out_features, self.in_features, self.in_features))
        else:
            W_offdiag = None

        def noise_chol():
            return jnp.exp(noise_logdiag)

        def W_chol():
            out = jnp.exp(W_logdiag)
            if self.parameterization == "dense":
                out = jnp.tril(W_offdiag, k=-1) + jnp.diag(out)
            return out

        def W():
            return DenseNormal(W_mean, W_chol())

        def noise():
            return Normal(noise_mean, noise_chol())

        def predictive(x):
            return (W() @ x[..., None]).squeeze(-1) + noise()

        def _get_train_loss_fn(x):
            def loss_fn(y, counter):
                W_instance = W()
                noise_instance = noise()
                pred_density = Normal(loc=(W_instance.mean @ x[..., None]).squeeze(-1), scale=noise_instance.scale)
                pred_likelihood = pred_density.log_prob(y)

                # Modify input x to account for empty elements in buffer
                x_mod = jnp.expand_dims(x * counter[:, None], -2)[..., None]
                trace_term = 0.5 * ((W_instance.covariance_weighted_inner_prod(x_mod, reduce_dim=False)) * noise_instance.precision)
                kl_term = KL(W_instance, self.prior_scale)
                wishart_term = (adjusted_dof * noise_instance.logdet_precision - 0.5 * self.wishart_scale * noise_instance.trace_precision)

                # Modify elbo to account for empty elements in buffer
                total_elbo = ((pred_likelihood.squeeze() - trace_term.squeeze()) * counter).sum() / counter.sum() # weighted loss
                total_elbo = total_elbo + self.regularization_weight * (wishart_term - kl_term)
                return - total_elbo
            return loss_fn

        def _get_val_loss_fn(x):
            def loss_fn(y):
                return -jnp.mean(predictive(x).log_prob(y))
            return loss_fn

        out = VBLLReturn(predictive(x), _get_train_loss_fn(x), _get_val_loss_fn(x))
        return out


class FifoVBLL(FifoSGD):
    def __init__(self, apply_fn, lossfn, tx, buffer_size, dim_features, dim_output, n_inner=1):
        super().__init__(apply_fn, lossfn, tx, buffer_size, dim_features, dim_output, n_inner)

    def sample_fn(self, key, bel):
        def fn(x):
            pp = self.predict_obs(bel, x)
            y_sampled = pp.predictive(rng_key=key)
            return y_sampled.squeeze()
        return fn

    def update(self, bel, y, x):
        return self.update_state(bel, x, y)

    def mean_fn(self, bel, x):
        pp = self.predict_obs(bel, x)
        return pp.predictive.mean

    def cov_fn(self, bel, x):
        pp = self.predict_obs(bel, x)
        return pp.predictive.covariance_diagonal
