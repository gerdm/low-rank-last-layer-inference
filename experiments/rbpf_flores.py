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


class LowRankLastLayerRBPF(flores.LowRankLastLayer):
    def __init__(self, mean_fn_tree, covariance_fn, rank, dynamics_hidden, dynamics_last, num_particles):
        super().__init__(mean_fn_tree, covariance_fn, rank, dynamics_hidden, dynamics_last)
        self.num_particles = num_particles
    
    def init_bel(self, params, cov_hidden, cov_last, low_rank_diag, key=314):
        cov_last = jnp.ones(self.num_particles) * cov_last
        vinit_bel = jax.vmap(super().init_bel, in_axes=(None, None, 0, None, None))
        state = vinit_bel(params, cov_hidden, cov_last, low_rank_diag, key)
        return state
    
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
