import jax
import chex
import einops
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

class LowRankLastLayerEnsemble(flores.LowRankLastLayer):
    def __init__(self, mean_fn_tree, covariance_fn, rank, dynamics_hidden, dynamics_last, num_particles):
        """
        FLoRES agent with ensemble of initial covariances.
        """
        super().__init__(mean_fn_tree, covariance_fn, rank, dynamics_hidden, dynamics_last)
        self.num_particles = num_particles
    
    def init_bel(self, params, cov_hidden, cov_last, low_rank_diag, key=314):
        vinit_bel = jax.vmap(super().init_bel, in_axes=(None, None, 0, None, None))
        bel_init = vinit_bel(params, cov_hidden, cov_last, low_rank_diag, key)
        return bel_init
    
    def predict(self, bel):
        return jax.vmap(super().predict)(bel)
    
    def update(self, bel, y, x):
        vupdate = jax.vmap(super().update, in_axes=(0, None, None))
        bel_update = vupdate(bel, y, x)
        return bel_update

    def sample_fn(self, key, bel):
        keys = jax.random.split(key, self.num_particles)
        params = jax.vmap(super().sample_params, in_axes=0)(keys, bel).squeeze()
        def fn(x):
            sample_fn = jax.vmap(self.mean_fn, in_axes=(None, 0, None))
            return sample_fn(bel.mean_hidden[0], params, x).mean(axis=0).squeeze()
        return fn


class LowRankLastLayerRBPF(flores.LowRankLastLayer):
    """
    RBPF for the last-layer
    """
    def __init__(self, mean_fn_tree, covariance_fn, rank, dynamics_hidden, dynamics_last, num_particles):
        super().__init__(mean_fn_tree, covariance_fn, rank, dynamics_hidden, dynamics_last)
        self.num_particles = num_particles
        self.pytree_vmap = FLoRESState(mean_last=0, loading_last=0, mean_hidden=None, loading_hidden=None)
    
    def init_bel(self, params, cov_hidden, cov_last, low_rank_diag, key=314):
        cov_last = jnp.ones(self.num_particles) * cov_last
        vinit_bel = jax.vmap(super().init_bel, in_axes=(None, None, 0, None, None))
        state = vinit_bel(params, cov_hidden, cov_last, low_rank_diag, key)
        return state

    def init_bel(self, params, cov_hidden=1.0, cov_last=1.0, low_rank_diag=True, key=314):
        self.rfn, self.mean_fn, init_params_hidden, init_params_last = self._initialise_flat_fn(self.mean_fn_tree, params)
        self.jac_hidden = jax.jacrev(self.mean_fn, argnums=0)
        self.jac_last = jax.jacrev(self.mean_fn, argnums=1)
        nparams_hidden = len(init_params_hidden)
        nparams_last = len(init_params_last)

        key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        key_hidden, key_params = jax.random.split(key)
        loading_hidden = self._init_low_rank(key_hidden, nparams_hidden, cov_hidden, low_rank_diag)
        loading_last = cov_last * jnp.eye(nparams_last) # TODO: make it low rank as well?

        # TODO: do we need this? Should be optional
        init_params_last = jax.random.normal(key_params, (self.num_particles, len(init_params_last))) * jnp.sqrt(cov_last)
        # init_params_last = einops.repeat(init_params_last, "m -> s m", s=self.num_particles)
        loading_last = einops.repeat(loading_last, "i j -> s i j", s=self.num_particles)
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
        vmap_update = jax.vmap(self._update, in_axes=(self.pytree_vmap, None, None))
        bel, err = vmap_update(bel, y, x)

        # Generalise RBPF step!
        ix_max = jnp.linalg.norm(err, axis=-1).argmin()
        bel = bel.replace(
            mean_hidden=bel.mean_hidden[ix_max],
            loading_hidden=bel.loading_hidden[ix_max],
        )
        return bel


    def sample_fn(self, key, bel):
        keys = jax.random.split(key, self.num_particles)
        params = jax.vmap(super().sample_params, in_axes=(0, self.pytree_vmap))(keys, bel).squeeze()
        def fn(x):
            sfn = jax.vmap(self.mean_fn, in_axes=(None, 0, None))
            return sfn(bel.mean_hidden, params, x).mean(axis=0).squeeze()
        return fn
