import jax
import chex
import einops
import distrax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
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
class FLoRESReplayState:
    """State of the online last-layer low-rank inference machine"""
    mean_last: chex.Array
    loading_last: chex.Array
    mean_hidden: chex.Array
    loading_hidden: chex.Array
    buffer_X: chex.Array
    buffer_Y: chex.Array
    counter: chex.Array
    num_obs: chex.Array
    runlength: chex.Array = 0
    log_posterior: chex.Array = 0.0


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


@chex.dataclass
class FloresIGammaState:
    """
    State of the online last-layer low-rank inference machine
    with unknown inverse gamma distribution
    """
    mean_last: chex.Array
    loading_last: chex.Array
    mean_hidden: chex.Array
    loading_hidden: chex.Array
    alpha: chex.Array
    beta: chex.Array


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

    def sample_predictive(self, key, bel, x):
        key_choice, key_sample = jax.random.split(key)
        ix = jax.random.choice(key_choice, self.num_particles)
        bel_sub = jax.tree.map(lambda x: x[ix], bel)
        return super().sample_predictive(key_sample, bel_sub, x)

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

        rho_vals = jax.random.normal(key_rho, (self.num_particles,)) * jnp.sqrt(cov_rho) - 5.0
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

        # inverse  multi-quadratic weight
        # wt = 1 / jnp.sqrt(1 + err ** 2 / 3.0)
        # log_pp = wt.squeeze() * jax.scipy.stats.norm.logpdf(err.squeeze(), loc=0.0, scale=jnp.sqrt(jnp.exp(bel.rho)).squeeze())

        log_pp = jax.scipy.stats.norm.logpdf(err.squeeze(), loc=0.0, scale=S_half.squeeze())

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
        key_resample, key_jitter = jax.random.split(key)
        indices = jax.random.categorical(key_resample, log_weights, shape=(self.num_particles,))
        bel = jax.tree.map(lambda x: x[indices], bel)
        # bel = bel.replace(
        #     rho=bel.rho + jax.random.normal(key_jitter, (self.num_particles,)) * 0.01
        # )

        log_weights = -jnp.log(self.num_particles)
        bel = bel.replace(log_weight=jnp.full(self.num_particles, log_weights))
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
        # choose_ix = jax.random.categorical(key_choice, bel.log_weight)
        sample_last = jax.vmap(super().sample_params, in_axes=(0, self.pytree_vmap))(keys, bel).squeeze()
        sample_hidden_single = bel.mean_hidden[choose_ix]
        sample_last_single = sample_last[choose_ix]
        def fn(x):
            return self.mean_fn(sample_hidden_single, sample_last_single, x).squeeze()
            # sfn = jax.vmap(self.mean_fn, in_axes=(0, 0, None))
            # samples = sfn(bel.mean_hidden, sample_last, x)
            # weights = jnp.exp(bel.log_weight)
            # return jnp.einsum("s...,s->...", samples, weights).squeeze()
        return fn


class LowRankLastLayerReplay(flores.LowRankLastLayer):
    """
    Low-rank last-layer RBPF with replay
    """
    def __init__(self, mean_fn_tree, cov_fn, rank, dynamics_hidden, dynamics_last, buffer_size=1, p_change=0.01, theshold=0.5):
        super().__init__(mean_fn_tree, cov_fn, rank, dynamics_hidden, dynamics_last)
        # self.pytree_vmap = FLowUnknownRtState(mean_last=0, loading_last=0, mean_hidden=0, loading_hidden=0, rho=0, key=None)
        self.pytree_vmap = 0
        self.buffer_size = buffer_size
        self.p_change = p_change
        self.threshold = theshold


    def init_bel(self, params, dim_in, dim_obs, cov_hidden=1.0, cov_last=1.0, low_rank_diag=True, key=314):
        bel_sup = super().init_bel(params, cov_hidden, cov_last, low_rank_diag, key)
        bel_replay = FLoRESReplayState(
            mean_last=bel_sup.mean_last,
            loading_last=bel_sup.loading_last,
            mean_hidden=bel_sup.mean_hidden,
            loading_hidden=bel_sup.loading_hidden,
            buffer_X=jnp.zeros((self.buffer_size, dim_in)),
            buffer_Y=jnp.zeros((self.buffer_size, dim_obs)),
            counter=jnp.zeros((self.buffer_size,)),
            num_obs=0,
        )
        self.init_mean_last = bel_sup.mean_last
        self.init_loading_last = bel_sup.loading_last
        self.init_mean_hidden = bel_sup.mean_hidden
        self.init_loading_hidden = bel_sup.loading_hidden
        return bel_replay


    def compute_log_posterior(self, y, X, bel, bel_prior):
        log_joint_increase = self.log_predictive_density(y, X, bel) + jnp.log1p(-self.p_change)
        log_joint_reset = self.log_predictive_density(y, X, bel_prior) + jnp.log(self.p_change)

        # Concatenate log_joints
        log_joint = jnp.array([log_joint_reset, log_joint_increase])
        log_joint = jnp.nan_to_num(log_joint, nan=-jnp.inf, neginf=-jnp.inf)
        # Compute log-posterior before reducing
        log_posterior_increase = log_joint_increase - jax.nn.logsumexp(log_joint)
        log_posterior_reset = log_joint_reset - jax.nn.logsumexp(log_joint)

        return log_posterior_increase, log_posterior_reset
    

    def deflate_belief(self, bel, bel_prior):
        gamma = jnp.exp(bel.log_posterior)

        new_mean_last = bel.mean_last * gamma  + (1 - gamma) * bel_prior.mean_last
        new_loading_last = bel.loading_last * gamma + (1 - gamma) * bel_prior.loading_last
        bel = bel.replace(mean_last=new_mean_last, loading_last=new_loading_last)
        return bel


    def log_predictive_density(self, y, X, bel):
        """
        compute the log-posterior predictive density
        of the moment-matched Gaussian
        """
        y = jnp.atleast_1d(y)
        log_p_pred = self.predictive_density(bel, X).log_prob(y).squeeze()
        return log_p_pred
    
    def _update_buffer(self, step, buffer, item):
        ix_buffer = step % self.buffer_size
        buffer = buffer.at[ix_buffer].set(item)
        return buffer
    
    def update_single(self, bel, y, x):
        bel_prior = bel.replace(
            mean_last=self.init_mean_last,
            loading_last=self.init_loading_last,
            mean_hidden=self.init_mean_hidden,
            loading_hidden=self.init_loading_hidden,
            runlength=0,
        )
        bel = self.deflate_belief(bel, bel_prior)
        log_posterior_increase, log_posterior_reset = self.compute_log_posterior(y, x, bel, bel_prior)
        bel = bel.replace(runlength=bel.runlength + 1, log_posterior=log_posterior_increase)

        err, gain_hidden, gain_last, J_hidden, J_last, R_half = self.innovation_and_gain(bel, y, x)

        mean_hidden, loading_hidden = self._update_hidden(bel, J_hidden, gain_hidden, R_half, err)
        mean_last, loading_last = self._update_last(bel, J_last, gain_last, R_half, err)

        posterior_increase = jnp.exp(log_posterior_increase)
        bel_prior = bel_prior.replace(log_posterior=log_posterior_reset)

        bel = bel.replace(
            mean_hidden=mean_hidden,
            mean_last=mean_last,
            loading_hidden=loading_hidden,
            loading_last=loading_last,
        )

        no_changepoint = posterior_increase >= self.threshold
        bel_update = jax.tree.map(
            lambda update, prior: update * no_changepoint + prior * (1 - no_changepoint),
            bel, bel_prior
        )

        return bel_update


    def update(self, bel, y, x):
        # Update buffers
        bel = bel.replace(num_obs=bel.num_obs + 1)
        X_update = self._update_buffer(bel.num_obs, bel.buffer_X, x)
        Y_update = self._update_buffer(bel.num_obs, bel.buffer_Y, y)
        counter_update = self._update_buffer(bel.num_obs, bel.counter, 1.0)
        bel = bel.replace(
            buffer_X=X_update,
            buffer_Y=Y_update,
            counter=counter_update
        )

        def _update_in_buffer(bel, xs):
            y, x, counter = xs
            bel_update = self.update_single(bel, y, x)
            bel = jax.lax.cond(counter==1.0, lambda: bel_update, lambda: bel)
            return bel, None
        
        xs = (bel.buffer_Y, bel.buffer_X, bel.counter)
        bel, _ = jax.lax.scan(_update_in_buffer, bel, xs)

        return bel


class LowRankLastLayerGamma(flores.LowRankLastLayer):
    """
    """
    def __init__(self, mean_fn_tree, rank, dynamics_hidden, dynamics_last):
        super().__init__(mean_fn_tree, lambda x: jnp.eye(1) * 1e-4, rank, dynamics_hidden, dynamics_last)


    def init_bel(
        self, params, num_arms,
        cov_hidden=1.0, cov_last=1.0,
        beta_prior=1.0, alpha_prior=1.0,
        low_rank_diag=True, key=314,
    ):
        bel_init_sup = super().init_bel(params, cov_hidden, cov_last, low_rank_diag, key)

        bel_init = FloresIGammaState(
            mean_last=bel_init_sup.mean_last,
            loading_last=bel_init_sup.loading_last,
            mean_hidden=bel_init_sup.mean_hidden,
            loading_hidden=bel_init_sup.loading_hidden,
            alpha=jnp.ones((num_arms,)) * alpha_prior,
            beta=jnp.ones((num_arms,)) * beta_prior,
        )

        return bel_init


    def innovation_and_gain(self, bel, y, x):
        arm = x[0].astype(int)
        yhat = self.mean_fn(bel.mean_hidden, bel.mean_last, x)
        R_half = jnp.atleast_2d(jnp.sqrt(bel.beta[arm] / (bel.alpha[arm] - 1)) * jnp.eye(1))
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

        return err, gain_hidden, gain_last, J_hidden, J_last, R_half
    
    def update(self, bel, y, x):
        err, gain_hidden, gain_last, J_hidden, J_last, R_half = self.innovation_and_gain(bel, y, x)

        mean_hidden, loading_hidden = self._update_hidden(bel, J_hidden, gain_hidden, R_half, err)
        mean_last, loading_last = self._update_last(bel, J_last, gain_last, R_half, err)

        # Update alpha and beta
        arm = x[0].astype(int)
        alpha_new = bel.alpha.at[arm].add(1/2)
        # sq_prev = jnp.linalg.solve(bel.loading_last.T @ bel.loading_last, bel.mean_last) @ bel.mean_last
        # sq_update = jnp.linalg.solve(loading_last.T @ loading_last, mean_last) @ mean_last
        beta_new_update = (err ** 2) / 2
        beta_new = bel.beta.at[arm].add(beta_new_update.squeeze())

        bel = bel.replace(
            mean_hidden=mean_hidden,
            mean_last=mean_last,
            loading_hidden=loading_hidden,
            loading_last=loading_last,
            alpha=alpha_new,
            beta=beta_new,
        )
        return bel


def compute_eff(Wt, x):
  """Computes (Wt.T @ Wt)^+ @ x efficiently."""
  M = Wt @ Wt.T                 # Small (d, d) matrix
  M_pinv = jnp.linalg.pinv(M)   # Pseudoinverse of M
  w = Wt.T @ (M_pinv @ M_pinv) @ (Wt @ x) # Combine steps
  return w


class VBLLFlores(flores.LowRankLastLayer):
    """
    Variational Bayes for Flores
    """
    def __init__(self, mean_fn_tree, loss_fn_tree, rank, dynamics_hidden, dynamics_last):
        super().__init__(mean_fn_tree, lambda x: jnp.eye(1) * 1e-4, rank, dynamics_hidden, dynamics_last)
        self.loss_fn_tree = loss_fn_tree

    def _initialise_lossfn(self, lossfn_tree, params):
        """
        Initialize ravelled function and gradients
        """
        last_layer_params = params["params"]["last_layer"]
        # dim_last_layer_params = len(ravel_pytree(last_layer_params)[0])

        _, rfn = ravel_pytree(params)

        @jax.jit
        def loss_fn(params_hidden, params_last, x, y):
            params = jnp.concat([params_hidden, params_last])
            return lossfn_tree(rfn(params), x, y).squeeze()


        return loss_fn

    def _initialise_flat_fn(self, apply_fn, params):
        """
        Initialize ravelled function and gradients
        """
        last_layer_params = params["params"]["last_layer"]
        dim_last_layer_params = len(ravel_pytree(last_layer_params)[0])

        flat_params, rfn = ravel_pytree(params)
        flat_params_last = flat_params[-dim_last_layer_params:]
        flat_params_hidden = flat_params[:-dim_last_layer_params]

        # @jax.jit
        def mean_fn(params_hidden, params_last, x):
            params = jnp.concat([params_hidden, params_last])
            return apply_fn(rfn(params), x)


        return rfn, mean_fn, flat_params_hidden, flat_params_last
    
    def init_bel(self, params, cov_hidden=1.0, cov_last=1.0, low_rank_diag=True, key=314):
        """
        Modified init function to include the loss function
        """
        self.lossfn = self._initialise_lossfn(self.loss_fn_tree, params)
        bel_init = super().init_bel(params, cov_hidden, cov_last, low_rank_diag, key)

        # loading_last = jax.random.orthogonal(jax.random.PRNGKey(31415), len(bel_init.mean_last))
        # bel_init = bel_init.replace(loading_last=loading_last)

        @jax.jit
        def sample_predictive(key, bel, x):
            def fn(x):
                pp = self.mean_fn(bel.mean_hidden, bel.mean_last, x)
                y_sampled = pp.predictive(rng_key=key)
                return y_sampled.squeeze()
            return fn(x)

        self.sample_predictive = sample_predictive       

        return bel_init


    def predict(self, bel):
        return bel

    def _update_hidden(self, bel, J):
        gain_matrix = [bel.loading_hidden, J[None, :]]
        loading_hidden = self.add_project(gain_matrix)
        gain = compute_eff(jnp.concat(gain_matrix, axis=0), J)
        mean_hidden = bel.mean_hidden - gain

        return mean_hidden, loading_hidden


    def _update_last(self, bel, J):
        gain_matrix = [bel.loading_last, J[None, :]]
        loading_last = self.add_sqrt(gain_matrix)
        gain = compute_eff(jnp.concat(gain_matrix, axis=0), J)
        mean_last = bel.mean_last - gain

        return mean_last, loading_last


    def update(self, bel, y, x):
        J_hidden = jax.grad(self.lossfn, argnums=0)(bel.mean_hidden, bel.mean_last, x, y)
        J_last = jax.grad(self.lossfn, argnums=1)(bel.mean_hidden, bel.mean_last, x, y)

        mean_hidden, loading_hidden = self._update_hidden(bel, J_hidden)
        mean_last, loading_last = self._update_last(bel, J_last)

        bel = bel.replace(
            mean_hidden=mean_hidden,
            mean_last=mean_last,
            loading_hidden=loading_hidden,
            loading_last=loading_last,
        )
        return bel
