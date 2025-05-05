import jax
import chex
import distrax
import jax.numpy as jnp
from functools import partial
from .base_filter import BaseFilter

@chex.dataclass
class LoFiState:
    """State of the Low Rank Filter"""
    mean: chex.Array
    diagonal: chex.Array
    low_rank: chex.Array


class LowRankPrecisionFilter(BaseFilter):
    def __init__(
        self, mean_fn, cov_fn, dynamics_covariance, rank, inflate_diag=True
    ):
        self.mean_fn_tree = mean_fn
        self.cov_fn = cov_fn
        self.dynamics_covariance = dynamics_covariance
        self.rank = rank
        self.inflate_diag = inflate_diag

    def init_bel(self, params, cov=1.0):
        self.rfn, self.mean_fn, init_params = self._initialise_flat_fn(self.mean_fn_tree, params)
        self.grad_mean = jax.jacrev(self.mean_fn)
        nparams = len(init_params)
        low_rank = jnp.zeros((nparams, self.rank))
        diagonal = jnp.ones(nparams) / cov # From covariance to precision term

        return LoFiState(
            mean=init_params,
            low_rank=low_rank,
            diagonal=diagonal,
        )

    def sample_params(self, key, bel):
        """
        Sample parameters from a low-rank variational Gaussian approximation.
        This implementation avoids the explicit construction of the
        (D x D) covariance matrix.

        We take s ~ N(0, W W^T + Psi I)

        Implementation based on §4.2.2 of the L-RVGA paper [1].

        [1] Lambert, Marc, Silvère Bonnabel, and Francis Bach.
        "The limited-memory recursive variational Gaussian approximation (L-RVGA).
         Statistics and Computing 33.3 (2023): 70.
        """
        key_x, key_eps = jax.random.split(key)
        dim_full, dim_latent = bel.low_rank.shape
        Psi_inv = 1 / bel.diagonal

        eps_sample = jax.random.normal(key_eps, (dim_latent,))
        x_sample = jax.random.normal(key_x, (dim_full,)) * jnp.sqrt(Psi_inv)

        I_latent = jnp.eye(dim_latent)
        # M = I + W^T Psi^{-1} W
        M = I_latent + jnp.einsum("ji,j,jk->ik", bel.low_rank, Psi_inv, bel.low_rank)
        # L = Psi^{-1} W^T M^{-1}
        L_tr = jnp.linalg.solve(M.T, jnp.einsum("i,ij->ji", Psi_inv, bel.low_rank))

        # samples = (I - LW^T)x + Le
        term1 = jnp.einsum("ji,kj->ik", L_tr, bel.low_rank)
        x_transform = jnp.einsum("ij,j->i", term1, x_sample)
        eps_transform = jnp.einsum("ji,j->i", L_tr, eps_sample)
        samples = x_sample + x_transform + eps_transform
        return samples + bel.mean


    def sample_fn(self, key, bel):
        params = self.sample_params(key, bel)
        def fn(x): return self.mean_fn(params, x).squeeze()
        return fn


    def predict_fn(self, bel, X):
        """
        Similar to self.mean_fn, but we pass the belief state (non-differentiable).
        This is useful for the case when we want to predict using different agents.
        """
        return self.mean_fn(bel.mean, X)


    def predictive_density(self, bel, X):
        """
        Equation (59) - (61)
        """
        mean = self.mean_fn(bel.mean, X).astype(float)
        Rt = jnp.atleast_2d(self.cov_fn(mean))

        Ht = self.grad_mean(bel.mean, X)

        diag_inverse = 1 / bel.diagonal
        C1 = jnp.einsum("ji,j,jk->ik", bel.low_rank, diag_inverse, bel.low_rank)
        C1 = jnp.linalg.inv(jnp.eye(self.rank) + C1)

        # Building these two terms explicitly is computationally expensive
        # C2 = jnp.einsum("i,ij,jk,lk,l->il", diag_inverse, bel.low_rank, C1, bel.low_rank, diag_inverse)
        # C3 = jnp.eye(len(bel.mean)) * diag_inverse  - C2
        # covariance = jnp.einsum("ij,jk,lk->il", Ht, C3, Ht) + Rt
        
        cov1 = jnp.einsum("ij,kj,j->ik", Ht, Ht, diag_inverse, optimize=True)
        cov2 = jnp.einsum("ai,i,ij,jk,lk,l,bl->ab", Ht, diag_inverse, bel.low_rank, C1, bel.low_rank, diag_inverse, Ht, optimize=True)
        covariance = cov1 - cov2 + Rt

        predictive = distrax.MultivariateNormalFullCovariance(mean, covariance)
        return predictive

    def sample_predictive(self, key, bel, x):
        dist = self.predictive_density(bel, x)
        sample = dist.sample(seed=key)
        return sample

    def predict(self, bel):
        I_lr = jnp.eye(self.rank)
        mean_pred = bel.mean
        diag_pred = 1 / (1 / bel.diagonal + self.dynamics_covariance)

        C = jnp.einsum("ji,j,jk->ik",
            bel.low_rank, (1 / bel.diagonal - diag_pred / bel.diagonal ** 2), bel.low_rank
        )
        C = jnp.linalg.inv(I_lr + C)
        cholC = jnp.linalg.cholesky(C)

        low_rank_pred = jnp.einsum(
            "i,i,ij,jk->ik",
            diag_pred, 1 / bel.diagonal, bel.low_rank, cholC
        )

        bel_pred = bel.replace(
            mean=mean_pred,
            diagonal=diag_pred,
            low_rank=low_rank_pred,
        )
        return bel_pred


    def _svd(self, W):
        """
        Fast implementation of reduced SVD

        See: https://math.stackexchange.com/questions/3685997/how-do-you-compute-the-reduced-svd
        """
        singular_vectors, singular_values, _ = jnp.linalg.svd(W.T @ W, full_matrices=False, hermitian=True)
        singular_values = jnp.sqrt(singular_values)
        singular_values_inv = jnp.where(singular_values != 0.0, 1 / singular_values, 0.0)
        singular_vectors = jnp.einsum("ij,jk,k->ik", W, singular_vectors, singular_values_inv)
        return singular_values, singular_vectors


    def _update_dlr(self, low_rank_hat):
        singular_values, singular_vectors = self._svd(low_rank_hat)

        singular_vectors_drop = singular_vectors[:, self.rank:] # Ut
        singular_values_drop = singular_values[self.rank:] # Λt

        # Update new low rank
        singular_vectors = singular_vectors[:, :self.rank] # Ut
        singular_values = singular_values[:self.rank] # Λt
        low_rank_new = jnp.einsum("Dd,d->Dd", singular_vectors, singular_values)

        # Obtain additive term for diagonal
        diag_drop = jnp.einsum(
            "ij,j,ij,j->i",
            singular_vectors_drop, singular_values_drop,
            singular_vectors_drop, singular_values_drop
        )

        return low_rank_new, diag_drop


    def update(self, bel_pred, y, x):
        yhat = self.mean_fn(bel_pred.mean, x)
        Rt = jnp.atleast_2d(self.cov_fn(y))
        Ht = self.grad_mean(bel_pred.mean, x)

        At = jnp.linalg.inv(jnp.linalg.cholesky(Rt))
        memory_entry = Ht.T @ At.T
        _, n_out = memory_entry.shape

        low_rank_hat = jnp.c_[bel_pred.low_rank, memory_entry]
        inverse_diag = 1 / bel_pred.diagonal
        Gt = jnp.linalg.pinv(
            jnp.eye(self.rank + n_out) +
            jnp.einsum("ji,j,jk->ik", low_rank_hat, inverse_diag, low_rank_hat)
        )

        err = y - yhat

        # LoFi gain times innovation
        K1 = jnp.einsum(
            "i,ji,kj,kl,l->i",
            inverse_diag, Ht, At, At, err
        )
        K2 = jnp.einsum(
            "i,ij,jk,lk,l,ml,nm,no,o->i",
            inverse_diag, low_rank_hat, Gt,
            low_rank_hat, inverse_diag,
            Ht, At, At, err
        )
        Kt_err = K1 - K2

        mean_new = bel_pred.mean + Kt_err
        low_rank_new, diag_drop = self._update_dlr(low_rank_hat)
        diag_new = bel_pred.diagonal + diag_drop * self.inflate_diag

        bel_new = bel_pred.replace(
            mean=mean_new,
            low_rank=low_rank_new,
            diagonal=diag_new,
        )
        return bel_new

