import jax
import chex
import distrax
import jax.numpy as jnp
from functools import partial
from jax.scipy.linalg import solve_triangular
from .base_filter import BaseFilter

@chex.dataclass
class LowRankState:
    """State of the Low Rank Filter"""
    mean: chex.Array
    low_rank: chex.Array
    epull: float =  0.0 # TODO: remove
    econs: float = 0.0
    q: float = 0.0

def orthogonal(key, n, m):
    """
    https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L2041-L2095
    """
    z = jax.random.normal(key, (max(n, m), min(n, m)))
    q, r = jnp.linalg.qr(z)
    d = jnp.linalg.diagonal(r)
    x = q * jnp.expand_dims(jnp.sign(d), -2)
    return x.T


class LowRankCovarianceFilter(BaseFilter):
    def __init__(
        self, mean_fn, cov_fn, dynamics_covariance, rank
    ):
        self.mean_fn_tree = mean_fn
        self.cov_fn = cov_fn
        self.dynamics_covariance = dynamics_covariance
        self.rank = rank

    def _init_low_rank(self, key, nparams, cov, diag):
        if diag:
            loading_hidden = cov * jnp.fill_diagonal(jnp.zeros((self.rank, nparams)), jnp.ones(nparams), inplace=False)
        else:
            # loading_hidden = cov * orthogonal(key, self.rank, nparams)
            key_Q, key_R = jax.random.split(key)
            A = jax.random.normal(key_Q, (self.rank, self.rank))
            Q, _ = jnp.linalg.qr(A)
            
            P = jax.random.normal(key_R, (self.rank, nparams))
            loading_hidden = Q @ P
            loading_hidden = loading_hidden / jnp.linalg.norm(loading_hidden, axis=-1, keepdims=True) * jnp.sqrt(cov)


        return loading_hidden

    def sample_params(self, key, bel, shape=None):
        """
        TODO: Double check!!
        """
        dim_full = len(bel.mean)
        shape = shape if shape is not None else (1,)
        shape_sub = (*shape, self.rank)
        eps = jax.random.normal(key, shape_sub)

        # params = jnp.einsum("ji,sj->si", bel.low_rank, eps) + eps_full * jnp.sqrt(self.dynamics_covariance) + bel.mean
        params = jnp.einsum("ji,sj->si", bel.low_rank, eps) + bel.mean
        return params

    def sample_fn(self, key, bel):
        params = self.sample_params(key, bel).squeeze()
        def fn(x): return self.mean_fn(params, x).squeeze()
        return fn

    def predictive_density(self, bel, x):
        yhat = self.mean_fn(bel.mean, x).astype(float)
        Rt = jnp.atleast_2d(self.cov_fn(yhat))
        Ht = self.grad_mean_fn(bel.mean, x)
        W = bel.low_rank

        C = jnp.r_[W @ Ht.T, jnp.sqrt(self.dynamics_covariance) * Ht.T]
        S = C.T @ C + Rt
        dist = distrax.MultivariateNormalFullCovariance(loc=yhat, covariance_matrix=S)
        return dist

    def sample_predictive(self, key, bel, x):
        dist = self.predictive_density(bel, x)
        sample = dist.sample(seed=key)
        return sample

    def init_bel(self, params, cov=1.0, low_rank_diag=True, key=314):
        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        self.rfn, self.mean_fn, init_params = self._initialise_flat_fn(self.mean_fn_tree, params)
        self.grad_mean_fn = jax.jacrev(self.mean_fn)
        nparams = len(init_params)
        low_rank = self._init_low_rank(key, nparams, cov, low_rank_diag)

        return LowRankState(
            mean=init_params,
            low_rank=low_rank,
        )

    def project(self, cst, *matrices):
        """
        Create rank-d matrix P such that
        P^T P approx A + B
        """
        Z = jnp.vstack(matrices)
        ZZ = jnp.einsum("ij,kj->ik", Z, Z)
        singular_vectors, singular_values, _ = jnp.linalg.svd(ZZ, hermitian=True, full_matrices=False)
        singular_values = jnp.sqrt(singular_values + cst) # square root of eigenvalues
        singular_values_inv = jnp.where(singular_values != 0.0, 1 / singular_values, 0.0)

        P = jnp.einsum("i,ji,jk->ik", singular_values_inv, singular_vectors, Z)
        P = jnp.einsum("d,dD->dD", singular_values[:self.rank], P[:self.rank])
        eigvals  = singular_values ** 2
        return P, eigvals

    def predict(self, bel):
        mean_pred = bel.mean
        low_rank_pred = bel.low_rank

        gamma = 1.0
        bel_pred = bel.replace(
            mean=gamma * mean_pred,
            low_rank=gamma * low_rank_pred,
        )
        return bel_pred
    
    def _innovation_and_gain(self, bel, y, x, q):
        yhat = self.mean_fn(bel.mean, x).astype(float)
        Rt_half = jnp.linalg.cholesky(jnp.atleast_2d(self.cov_fn(yhat)), upper=True)
        Ht = self.grad_mean_fn(bel.mean, x)
        W = bel.low_rank

        C = jnp.r_[W @ Ht.T, jnp.sqrt(q) * Ht.T, Rt_half]
        # C = jnp.r_[W @ Ht.T, Rt_half]
        S_half = jnp.linalg.qr(C, mode="r") # Squared-root of innovation

        # transposed Kalman gain and innovation
        Mt = solve_triangular(S_half, solve_triangular(S_half.T, Ht, lower=True), lower=False)
        Kt_T = Mt @ W.T @ W + Mt * q
        err = y - yhat
        return Kt_T, err, Rt_half, Ht


    def predict_fn(self, bel, X):
        """
        Similar to self.mean_fn, but we pass the belief state (non-differentiable).
        This is useful for the case when we want to predict using different agents.
        """
        return self.mean_fn(bel.mean, X)
    

    def update(self, bel, y, x):
        q = self.dynamics_covariance
        Kt_T, err, Rt_half, Ht = self._innovation_and_gain(bel, y, x, q)
        mean_update = bel.mean + jnp.einsum("ij,i->j", Kt_T, err)


        low_rank_update, eigvals = self.project(q,
            bel.low_rank - bel.low_rank @ Ht.T @ Kt_T,
            Rt_half @ Kt_T
        )

        vnorm = jnp.sqrt(jnp.einsum("ji,jk,lk,li->", Kt_T, Ht, Ht, Kt_T, optimize=True))
        eps_pull = (2 * vnorm + vnorm ** 2)
        eps_trunc = jnp.sqrt(jnp.sum(eigvals[self.rank:]**2))
        # norm_tilde = jnp.sqrt(jnp.sum(eigvals**2))

        # rho = 0.1
        # q = (rho * norm_tilde - eps_trunc) / epull
        # q = jnp.clip(q, 0.0, 0.5)

        epull = q * eps_pull
        econs = eps_trunc
        # relative_fro_error = epull + econs

        bel = bel.replace(
            mean=mean_update,
            low_rank=low_rank_update,
            epull=epull,
            econs=econs,
            q=q
        )
        return bel

