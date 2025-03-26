import jax
import chex
import einops
import distrax
import jax.numpy as jnp
from rebayes_mini.methods import low_rank_last_layer as flores


@chex.dataclass
class KFState:
    mean: jax.Array
    cov: jax.Array
    regime: jax.Array # KF config

    @staticmethod
    def init(key, mean, cov, n_particles, n_configs):
        means = einops.repeat(mean, "m -> s m", s=n_particles)
        covs = einops.repeat(cov, "i j -> s i j", s=n_particles)
        regimes = jax.random.choice(key, n_configs, (n_particles,)) # uniform sample over configs
        
        return KFState(
            mean=means, 
            cov=covs,
            regime=regimes,
        )

@chex.dataclass
class FLoRESState:
    """State of the online last-layer low-rank inference machine"""
    mean_last: chex.Array
    loading_last: chex.Array
    mean_hidden: chex.Array
    loading_hidden: chex.Array


@chex.dataclass
class FLowUnknownRtState:
    """State of the online last-layer low-rank inference machine"""
    mean_last: chex.Array
    loading_last: chex.Array
    mean_hidden: chex.Array
    loading_hidden: chex.Array
    rho: chex.Array
    key: jax.random.PRNGKey
    log_weight: chex.Array


class LowRankLastLayerEnsemble(flores.LowRankLastLayer):
    """
    Low-rank last-layer ensemble
    """
    def __init__(self, mean_fn_tree, covariance_fn, rank, dynamics_hidden, dynamics_last, num_particles):
        super().__init__(mean_fn_tree, covariance_fn, rank, dynamics_hidden, dynamics_last)
        self.num_particles = num_particles
        # self.pytree_vmap = FLoRESState(mean_last=0, loading_last=0, mean_hidden=None, loading_hidden=None)
        self.pytree_vmap = 0
    
    def init_bel(self, params, cov_hidden=1.0, cov_last=1.0, low_rank_diag=True, key=314):
        self.rfn, self.mean_fn, init_params_hidden, init_params_last = self._initialise_flat_fn(self.mean_fn_tree, params)
        self.jac_hidden = jax.jacrev(self.mean_fn, argnums=0)
        self.jac_last = jax.jacrev(self.mean_fn, argnums=1)
        nparams_hidden = len(init_params_hidden)
        nparams_last = len(init_params_last)

        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        key_hidden, key_params = jax.random.split(key)
        loading_hidden = self._init_low_rank(key_hidden, nparams_hidden, cov_hidden, low_rank_diag)
        loading_last = cov_last * jnp.eye(nparams_last)

        # TODO: do we need this? Should be optional
        init_params_last = jax.random.normal(key_params, (self.num_particles, len(init_params_last))) * jnp.sqrt(cov_last)
        # init_params_last = einops.repeat(init_params_last, "m -> s m", s=self.num_particles)
        loading_last = einops.repeat(loading_last, "i j -> s i j", s=self.num_particles)
        # Hidden state
        loading_hidden = einops.repeat(loading_hidden, "i j -> s i j", s=self.num_particles)
        init_params_hidden = einops.repeat(init_params_hidden, "m -> s m", s=self.num_particles)
        return FLoRESState(
            mean_last=init_params_last,
            loading_last=loading_last,
            mean_hidden=init_params_hidden,
            loading_hidden=loading_hidden,
        )
    
    def predict(self, bel):
        # TODO: vmap over self.pytree_vmap
        return bel
        # return jax.vmap(super().predict)(bel)
        
    def _update(self, bel, y, x):
        err, gain_hidden, gain_last, J_hidden, J_last, R_half = self.innovation_and_gain(bel, y, x)

        mean_hidden, loading_hidden = self._update_hidden(bel, J_hidden, gain_hidden, R_half, err)
        mean_last, loading_last = self._update_last(bel, J_last, gain_last, R_half, err)

        bel = bel.replace(
            mean_hidden=mean_hidden,
            mean_last=mean_last,
            loading_hidden=loading_hidden,
            loading_last=loading_last,
        )
        return bel, err

    def update(self, bel, y, x):
        # TODO: this should be the RBPF update step
        vmap_update = jax.vmap(self._update, in_axes=(self.pytree_vmap, None, None))
        bel, err = vmap_update(bel, y, x)

        return bel


    def sample_fn(self, key, bel):
        key_sample, key_choice = jax.random.split(key)
        keys = jax.random.split(key_sample, self.num_particles)
        choose_ix = jax.random.choice(key_choice, self.num_particles)
        sample_last = jax.vmap(super().sample_params, in_axes=(0, self.pytree_vmap))(keys, bel).squeeze()
        sample_hidden_single = bel.mean_hidden[choose_ix]
        sample_last_single = sample_last[choose_ix]
        def fn(x):
            return self.mean_fn(sample_hidden_single, sample_last_single, x).squeeze()
            # sfn = jax.vmap(self.mean_fn, in_axes=(None, 0, None))
            # return sfn(bel.mean_hidden, sample_last, x).mean(axis=0).squeeze()
        return fn

class LowRankLastLayerRBPF(flores.LowRankLastLayer):
    """
    RBPF for the last-layer with unknown observation noise (Rt)
    """
    def __init__(self, mean_fn_tree, rank, dynamics_hidden, dynamics_last, sigma_rho, num_particles, resample_threshold=None):
        super().__init__(mean_fn_tree, lambda x: x, rank, dynamics_hidden, dynamics_last)
        self.num_particles = num_particles
        self.sigma_rho = sigma_rho
        self.resample_threshold = resample_threshold if resample_threshold is not None else self.num_particles / 3
        # self.pytree_vmap = FLowUnknownRtState(mean_last=0, loading_last=0, mean_hidden=0, loading_hidden=0, rho=0, key=None)
        self.pytree_vmap = 0
    
    def init_bel(self, params, cov_hidden=1.0, cov_last=1.0, cov_rho=0.1, low_rank_diag=True, key=314):
        self.rfn, self.mean_fn, init_params_hidden, init_params_last = self._initialise_flat_fn(self.mean_fn_tree, params)
        self.jac_hidden = jax.jacrev(self.mean_fn, argnums=0)
        self.jac_last = jax.jacrev(self.mean_fn, argnums=1)
        nparams_hidden = len(init_params_hidden)
        nparams_last = len(init_params_last)

        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        key_hidden, key_params, key_rho, key_carry = jax.random.split(key, 4)
        loading_hidden = self._init_low_rank(key_hidden, nparams_hidden, cov_hidden, low_rank_diag)
        loading_last = cov_last * jnp.eye(nparams_last)

        # TODO: do we need this? Should be optional
        # init_params_last = jax.random.normal(key_params, (self.num_particles, len(init_params_last))) * jnp.sqrt(cov_last)
        init_params_last = einops.repeat(init_params_last, "m -> s m", s=self.num_particles)
        loading_last = einops.repeat(loading_last, "i j -> s i j", s=self.num_particles)
        # Hidden state
        loading_hidden = einops.repeat(loading_hidden, "i j -> s i j", s=self.num_particles)
        init_params_hidden = einops.repeat(init_params_hidden, "m -> s m", s=self.num_particles)

        rho_vals = jax.random.normal(key_rho, (self.num_particles,)) * jnp.sqrt(cov_rho)
        log_weights = jnp.zeros(self.num_particles)

        keys_carry = jax.random.split(key_carry, self.num_particles)
        return FLowUnknownRtState(
            mean_last=init_params_last,
            loading_last=loading_last,
            mean_hidden=init_params_hidden,
            loading_hidden=loading_hidden,
            rho=rho_vals,
            key=keys_carry,
            log_weight=log_weights,
        )

    def innovation_and_gain(self, bel, y, x):
        yhat = self.mean_fn(bel.mean_hidden, bel.mean_last, x)
        # TODO: Only supports one-dimensional observations: generalise to multi-dimensional
        R_half = jnp.sqrt(jnp.exp(bel.rho)) * jnp.eye(1)

        # Jacobian for hidden and last layer
        J_hidden = self.jac_hidden(bel.mean_hidden, bel.mean_last, x)
        J_last = self.jac_last(bel.mean_hidden, bel.mean_last, x)

        # Innovation
        err = y - yhat

        # Upper-triangular cholesky decomposition of the innovation
        S_half = self.add_sqrt([bel.loading_hidden @ J_hidden.T, bel.loading_last @ J_last.T, R_half])

        # Transposed gain matrices
        M_hidden = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, J_hidden))
        M_last = jnp.linalg.solve(S_half, jnp.linalg.solve(S_half.T, J_last))

        gain_hidden = M_hidden @ bel.loading_hidden.T @ bel.loading_hidden + M_hidden * self.dynamics_hidden
        gain_last = M_last @ bel.loading_last.T @ bel.loading_last

        # log_pp = distrax.MultivariateNormalTri(scale_tri=S_half, is_lower=False).log_prob(jnp.atleast_1d(err))
        # log_pp = distrax.Normal(loc=0, scale=jnp.sqrt(jnp.exp(bel.rho))).log_prob(err).squeeze()
        log_pp = jax.scipy.stats.norm.logpdf(err.squeeze(), loc=0.0, scale=jnp.sqrt(jnp.exp(bel.rho)).squeeze())

        return err, log_pp, gain_hidden, gain_last, J_hidden, J_last, R_half

    
    def predict(self, bel):
        # TODO: vmap over self.pytree_vmap
        return bel
        # return jax.vmap(super().predict)(bel)
        
    def _update(self, bel, y, x):
        # Update value for rho
        bel = bel.replace(
            rho=bel.rho + jax.random.normal(bel.key) * self.sigma_rho
        )
        err, log_pp, gain_hidden, gain_last, J_hidden, J_last, R_half = self.innovation_and_gain(bel, y, x)

        mean_hidden, loading_hidden = self._update_hidden(bel, J_hidden, gain_hidden, R_half, err)
        mean_last, loading_last = self._update_last(bel, J_last, gain_last, R_half, err)

        bel = bel.replace(
            mean_hidden=mean_hidden,
            mean_last=mean_last,
            loading_hidden=loading_hidden,
            loading_last=loading_last,
        )
        return bel, log_pp

    def _resample(self, key, log_weights, bel):
        indices = jax.random.categorical(key, log_weights, shape=(self.num_particles,))
        bel = jax.tree.map(lambda x: x[indices], bel)
        bel = bel.replace(log_weight=jnp.zeros(self.num_particles))
        return bel
    

    def _update_log_weight(self, key, log_weights, bel):
        bel = bel.replace(
            log_weight=log_weights
        )
        return bel


    def update(self, bel, y, x):
        # RBPF step
        vmap_update = jax.vmap(self._update, in_axes=(self.pytree_vmap, None, None))
        bel, log_weights = vmap_update(bel, y, x)
        # Sample with resampling
        keys = jax.vmap(jax.random.split, out_axes=0)(bel.key)
        key_resample = keys[0, 0]
        keys_new = keys[:, 1]

        log_weight = bel.log_weight + log_weights
        log_weight = log_weight - jax.nn.logsumexp(log_weight)

        ess = 1 / jnp.sum(jnp.exp(log_weight) ** 2)
        bel = jax.lax.cond(
            ess <= self.resample_threshold, 
            self._resample,
            self._update_log_weight,
            key_resample, log_weight, bel
        )
        bel = bel.replace(key=keys_new) # do not resample the key
        return bel


    def sample_fn(self, key, bel):
        key_sample, key_choice = jax.random.split(key)
        keys = jax.random.split(key_sample, self.num_particles)
        choose_ix = jax.random.choice(key_choice, self.num_particles)
        sample_last = jax.vmap(super().sample_params, in_axes=(0, self.pytree_vmap))(keys, bel).squeeze()
        sample_hidden_single = bel.mean_hidden[choose_ix]
        sample_last_single = sample_last[choose_ix]
        def fn(x):
            return self.mean_fn(sample_hidden_single, sample_last_single, x).squeeze()
            # sfn = jax.vmap(self.mean_fn, in_axes=(None, 0, None))
            # return sfn(bel.mean_hidden, sample_last, x).mean(axis=0).squeeze()
        return fn
