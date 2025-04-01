import jax
import optax
import numpy as np
import pandas as pd
import flax.linen as nn
import jax.numpy as jnp
from time import time
from functools import partial
from rbpf_flores import LowRankLastLayerGamma
from rbpf_flores import LowRankLastLayerReplay
from rbpf_flores import VBLLFlores
from rebayes_mini.methods import low_rank_last_layer as flores
from vbll_fifo import Regression, FifoVBLL, RegressionRefac

path = "../../KuaiRec 2.0/data/arms_05_raw.pkl"
df_all = pd.read_pickle(path).dropna()
df_all = df_all.sort_values("time_0")
df_all = df_all.sample(frac=1.0, random_state=31, replace=False)


target_cols = ["like_cnt", "share_cnt", "play_cnt", "play_duration", "comment_cnt"]
def in_target_cols(column):
    return any([target_col in column for target_col in target_cols])

# We start by building a bandit problem where we are only given the possible arms to pull and the (latent) rewards
X = df_all[[c for c in df_all if ("video_id" in c) or in_target_cols(c)]]
Y = df_all[[c for c in df_all if "watch_ratio" in c]]

X, Y = jax.tree.map(jnp.array, (X, Y))
n_obs = len(X)
n_arms = 5
n_videos = len(df_all["video_id_0"].unique())
n_features = len(target_cols) + 1 # video id and like_cnt

X = X.reshape(-1, n_arms, n_features)

X = X.at[..., 1:].set(jnp.log1p(X[..., 1:]))


class FeatureExtractor(nn.Module):
    """
    Encoder
    """
    n_videos: int
    embedding_dim: int
    dense_dim: int
    n_hidden: int

    @nn.compact
    def __call__(self, x):
        x_embedding = nn.Embed(self.n_videos, self.embedding_dim)(x[..., 0].astype(int))
        x_features = nn.Dense(self.dense_dim)(x[..., 1:])
        x = jnp.concat([x_embedding, x_features], axis=-1)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.elu(x)
        return x


class MLP(nn.Module):
    n_videos: int
    embedding_dim: int
    dense_dim: int
    n_hidden: int

    @nn.compact
    def __call__(self, x):
        x = FeatureExtractor(self.n_videos, self.embedding_dim, self.dense_dim, self.n_hidden)(x)
        x = nn.Dense(1, name="last_layer")(x) # reward
        return x


def step(state, xs, agent):
    bel, key = state
    yt, xt = xs
    n_arms = xt.shape[0]
    key_sample, key_update = jax.random.split(key, 2)

    bel = agent.predict(bel)

    keys_sample = jax.random.split(key_sample, n_arms)
    rewards_est = jax.vmap(agent.sample_predictive, in_axes=(0, None, 0))(keys_sample, bel, xt)
    
    # Choose the arm to pull
    arm = rewards_est.argmax()
    
    reward_obs = yt[arm]
    x_pulled = xt[arm]

    bel_update = agent.update(bel, reward_obs, x_pulled)
    state_update = (bel_update, key_update)
    return state_update, reward_obs


class FVBLLMLP(nn.Module):
    n_videos: int
    embedding_dim: int
    dense_dim: int
    n_hidden: int
    wishart_scale: float = 0.01
    regularization_weight: float = 1.0


    @nn.compact
    def __call__(self, x):
        x = FeatureExtractor(self.n_videos, self.embedding_dim, self.dense_dim, self.n_hidden)(x)
        x = RegressionRefac(
            in_features=self.n_hidden, out_features=1,
            wishart_scale=self.wishart_scale,
            regularization_weight=self.regularization_weight,
            name="last_layer",
        )(x)
        return x


### SETUP ####
key = jax.random.PRNGKey(3141)
key_init, key_run = jax.random.split(key)
n_obs = 1000

# Oracle
rewards_oracle = Y[:n_obs].max(axis=1)
print(rewards_oracle.sum())

### VBLL-Flores ####

vbl_mlp = FVBLLMLP(n_videos=n_videos, embedding_dim=10, n_hidden=10, dense_dim=10)
params_init_vbll = vbl_mlp.init(key_init, X[0,0])


def vbll_loss(params, x, y):
    res = vbl_mlp.apply(params, x)
    loss = res.train_loss_fn(y)
    return jnp.atleast_1d(loss)

agent = VBLLFlores(
    vbl_mlp.apply,
    vbll_loss,
    rank=20,
    dynamics_hidden=0.0,
    dynamics_last=0.0
)
bel_init = agent.init_bel(params_init_vbll)

print("*" * 30)
print("Running agent with Flores+VBLL")
bel_init = agent.init_bel(params_init_vbll)

time_init = time()
state_init = (bel_init, key_run)
XS = jnp.log1p(Y[:n_obs]), X[:n_obs]
_step = partial(step, agent=agent)
(bel_final_flores, _), rewards_vbll_flores = jax.lax.scan(_step, state_init, XS)
rewards_vbll_flores = np.exp(np.array(rewards_vbll_flores)) - 1
print(rewards_vbll_flores[-50:])
time_end = time()

print(rewards_vbll_flores.sum())
print(f"Running time {time_end - time_init:0.2f}s", end="\n" * 2)


#### VBLL #####

class VBLLMLP(nn.Module):
    n_videos: int
    embedding_dim: int
    dense_dim: int
    n_hidden: int
    wishart_scale: float = 0.01
    regularization_weight: float = 1.0


    @nn.compact
    def __call__(self, x):
        x = FeatureExtractor(self.n_videos, self.embedding_dim, self.dense_dim, self.n_hidden)(x)
        x = Regression(
            in_features=self.n_hidden, out_features=1,
            wishart_scale=self.wishart_scale,
            regularization_weight=self.regularization_weight,
            name="last_layer",
        )(x)
        return x

learning_rate = 1e-4
buffer_size = 1
n_inner = 1
vbl_mlp = VBLLMLP(n_videos=n_videos, embedding_dim=10, n_hidden=10, dense_dim=10)
params_init_vbll = vbl_mlp.init(key_init, X[0,0])

def lossfn(params, counter, x, y, apply_fn):
    res = apply_fn(params, x)
    return res.train_loss_fn(y, counter)

dim = X.shape[-1]

agent = FifoVBLL(
    vbl_mlp.apply,
    lossfn,
    tx=optax.adamw(learning_rate),
    buffer_size=buffer_size,
    dim_features=dim,
    dim_output=1,
    n_inner=n_inner,
)


print("Running agent with VBLL")
bel_init = agent.init_bel(params_init_vbll)

time_init = time()
state_init = (bel_init, key_run)
XS = jnp.log1p(Y[:n_obs]), X[:n_obs]
_step = partial(step, agent=agent)
(bel_final_fifo, _), rewards_vbll_fifo = jax.lax.scan(_step, state_init, XS)
rewards_vbll_fifo = np.exp(np.array(rewards_vbll_fifo)) - 1
time_end = time()

print(rewards_vbll_fifo.sum())
print(f"Running time {time_end - time_init:0.2f}")


### Flores ####

key = jax.random.PRNGKey(3141)
mlp = MLP(n_videos=n_videos, embedding_dim=10, dense_dim=10, n_hidden=10)
key_init, key_run = jax.random.split(key)
params_init = mlp.init(key_init, X[0, 0])
    

def cov_fn(y): return jnp.eye(1) * 1e-4
agent = flores.LowRankLastLayer(mlp.apply, cov_fn, rank=5, dynamics_hidden=0.0, dynamics_last=1e-4)


print("Running agent with Flores")
bel_init = agent.init_bel(params_init, cov_hidden=1.0, cov_last=1.0, low_rank_diag=True)
state_init = (bel_init, key_run)
XS = (jnp.log1p(Y[:n_obs]), X[:n_obs])
_step = partial(step, agent=agent)
time_init = time()
(bel_final, _), rewards_flores = jax.lax.scan(_step, state_init, XS)
rewards_flores = np.exp(np.array(rewards_flores)) - 1
time_end = time()

print(rewards_flores.sum())
print(f"Running time {time_end - time_init:.2f}", end=2*"\n")

import pdb; pdb.set_trace()