import jax
import chex
import einops
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
        