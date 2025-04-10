"""
Source: https://github.com/VectorInstitute/vbll/blob/main/vbll/jax/layers/regression.py

replay-SGD-compatible VBLL
"""
import jax
import distrax
import jax.numpy as jnp
import flax.linen as nn
from jax.flatten_util import ravel_pytree
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


class RegressionRefac(nn.Module):
    in_features: int
    out_features: int
    regularization_weight: float
    parameterization: str = "dense"
    prior_scale: float = 1.0
    wishart_scale: float = 1e-2
    dof: float = 1.0

    def setup(self):
        self.adjusted_dof = (self.dof + self.out_features + 1.) / 2.
        self.noise_mean = self.param("noise_mean", nn.initializers.zeros, (self.out_features,))
        self.noise_logdiag = self.param("noise_logdiag", nn.initializers.normal(), (self.out_features,))
        self.W_mean = self.param("W_mean", nn.initializers.normal(), (self.out_features, self.in_features))
        self.W_logdiag = self.param("W_logdiag", nn.initializers.normal(), (self.out_features, self.in_features))
        if self.parameterization == "dense":
            self.W_offdiag = self.param("W_offdiag", lambda rng, shape: jax.random.normal(rng, shape) / self.in_features, (self.out_features, self.in_features, self.in_features))
        else:
            self.W_offdiag = None


    def noise_chol(self):
        return jnp.exp(self.noise_logdiag)

    def W_chol(self):
        out = jnp.exp(self.W_logdiag)
        if self.parameterization == "dense":
            out = jnp.tril(self.W_offdiag, k=-1) + jnp.diag(out)
        return out

    def W(self):
        return DenseNormal(self.W_mean, self.W_chol())

    def noise(self):
        return Normal(self.noise_mean, self.noise_chol())

    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()


    def _get_train_loss_fn(self, x):
        def loss_fn(y, counter):
            W_instance = self.W()
            noise_instance = self.noise()
            pred_density = Normal(loc=(W_instance.mean @ x[..., None]).squeeze(-1), scale=noise_instance.scale)
            pred_likelihood = pred_density.log_prob(y)

            # Modify input x to account for empty elements in buffer
            # x_mod = jnp.expand_dims(x, -2)[..., None]
            x_mod = jnp.expand_dims(x * counter[:, None], -2)[..., None]
            trace_term = 0.5 * ((W_instance.covariance_weighted_inner_prod(x_mod, reduce_dim=False)) * noise_instance.precision)
            kl_term = KL(W_instance, self.prior_scale)
            wishart_term = (self.adjusted_dof * noise_instance.logdet_precision - 0.5 * self.wishart_scale * noise_instance.trace_precision)

            # Modify elbo to account for empty elements in buffer
            # total_elbo = ((pred_likelihood.squeeze() - trace_term.squeeze())).mean() # weighted loss
            total_elbo = ((pred_likelihood.squeeze() - trace_term.squeeze()) * counter).sum() / counter.sum() # weighted loss
            total_elbo = total_elbo + self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo
        return loss_fn


    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            return -jnp.mean(self.predictive(x).log_prob(y))
        return loss_fn


    @nn.compact
    def __call__(self, x, y=None):
        out = VBLLReturn(self.predictive(x), self._get_train_loss_fn(x), self._get_val_loss_fn(x))
        # out = self.predictive(x).log_prob(y)
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

    def sample_predictive(self, key, bel, x):
        fn = self.sample_fn(key, bel)
        return fn(x).squeeze()

    def predict(self, bel):
        return bel

    def update(self, bel, y, x):
        return self.update_state(bel, x, y)

    def mean_fn(self, bel, x):
        pp = self.predict_obs(bel, x)
        return pp.predictive.mean

    def cov_fn(self, bel, x):
        pp = self.predict_obs(bel, x)
        return pp.predictive.covariance_diagonal


class FifoLaplaceDiag(FifoSGD):
    """
    TODO: rename to FifoLaplaceReg
    """
    def __init__(self, apply_fn, lossfn, tx, buffer_size, dim_features, dim_output, n_inner=1):
        super().__init__(apply_fn, lossfn, tx, buffer_size, dim_features, dim_output, n_inner)

    def sample_fn(self, key, bel):
        # Sample from last-layer params
        params_sample = self.sample_params(key, bel)
        def fn(x):
            return self.apply_fn(params_sample, x).squeeze()
        return fn

    def predict(self, bel):
        return bel
    

    def _get_hessian_means(self, bel):
        params = jax.tree.map(jnp.copy, bel.params)
        params_all_flat, rfn_all = ravel_pytree(params)
        params_last = params["params"].pop("last_layer")
        params_last_flat, rfn = ravel_pytree(params_last)
        params_hidden_flat, rfn_hidden = ravel_pytree(params["params"])

        def lossfn(params_last, params_hidden, counter, X, y):
            params = {
                "params": {
                    "last_layer": rfn(params_last),
                    **params_hidden,
                }
            }
            return self.lossfn(params, counter, X, y, self.apply_fn).squeeze()
        
        # vhessian = jax.vmap(jax.hessian(lossfn, argnums=0), in_axes=(None, None, 0, 0, 0))
        vgrad = jax.vmap(jax.grad(lossfn, argnums=0), in_axes=(None, None, 0, 0, 0))
        # hessian = vhessian(params_last_flat, params["params"], bel.counter, bel.buffer_X, bel.buffer_y)
        grad = vgrad(params_last_flat, params["params"], bel.counter, bel.buffer_X, bel.buffer_y)
        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = jnp.linalg.norm(grad, axis=-1, keepdims=True)
        grad = grad / (grad_norm + 1e-6)  # Normalize gradients to have unit norm

        fisher = jnp.einsum("ij,ik->ijk", grad, grad)
        F = (fisher * bel.counter[:, None, None]).sum(axis=0) / bel.counter.sum()+ 1e-4 * jnp.eye(fisher.shape[-1])
        return params_last_flat, params_hidden_flat, rfn_all, F


    def get_posterior_predictive(self, bel, x):
        params = jax.tree.map(jnp.copy, bel.params)
        params_last = params["params"].pop("last_layer")
        params_last_flat, rfn_last = ravel_pytree(params_last)


        def apply_fn(params_last, params_hidden, x):
            params = {
                "params": {
                    "last_layer": rfn_last(params_last),
                    **params_hidden,
                }
            }
            res =  self.apply_fn(params, x).squeeze()
            return res
        
        *_, H = self._get_hessian_means(bel)

        grad_last = jax.jacrev(apply_fn, argnums=0)(params_last_flat, params["params"], x)
        map_est = apply_fn(params_last_flat, params["params"], x)
        map_est = jax.nn.softmax(map_est)
        # TODO: this should be implemented in a child class (or given by the user)
        Rt = jnp.diag(map_est) - jnp.outer(map_est, map_est)
        cov = jnp.einsum("ij,jk,lk->il", grad_last, H, grad_last) + 0.1 * jnp.eye(grad_last.shape[0]) + Rt
        return distrax.MultivariateNormalFullCovariance(loc=map_est, covariance_matrix=cov)


    def sample_params(self, key, bel):
        params_last_flat, params_hidden_flat, rfn_all, H = self._get_hessian_means(bel)

        # Compute cholesky decomposition
        # U = jnp.linalg.cholesky(H).T
        # Compute SVD of the Hessian
        U, _, _ = jnp.linalg.svd(H)  # Only need U for sampling

        # Sample from standard normal
        z = jax.random.normal(key, shape=params_last_flat.shape)
        # Compute sample
        params_last_sample = params_last_flat + jnp.linalg.solve(U.T, z)
        # concatanate sampled params with the rest of the params
        params_sample = jnp.concatenate([params_hidden_flat, params_last_sample])
        # rebund the params
        params_sample = rfn_all(params_sample)
        return params_sample

    def sample_predictive(self, key, bel, x):
        dist = self.get_posterior_predictive(bel, x)
        sample = dist.sample(seed=key)
        return sample

    def sample_predictive_random(self, key, bel, x, n_samples=100, sigma2=1e-2):
        # Split the key into n_samples keys.
        keys = jax.random.split(key, n_samples)
        
        def one_sample_predictive(k):
            key_params, key_noise = jax.random.split(k)
            params_sample = self.sample_params(key_params, bel)
            # Compute the predictive mean via forward pass.
            mu = self.apply_fn(params_sample, x)
            noise = jax.random.normal(key_noise, shape=mu.shape) * jnp.sqrt(sigma2)
            return mu + noise
        
        # Vectorize the sampling procedure.
        predictions = jax.vmap(one_sample_predictive)(keys)
        return predictions.mean(axis=0)

    
    def update(self, bel, y, x):
        return self.update_state(bel, x, y)