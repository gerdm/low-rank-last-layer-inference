import jax
import gymnax
import flax.linen as nn
import jax.numpy as jnp
from functools import partial
from rebayes_mini.methods import low_rank_last_layer as flores
from rebayes_mini.methods import low_rank_filter as lofi


class CNN(nn.Module):
    num_actions: int = 10

    @nn.compact
    def __call__(self, x):
        x = x if len(x.shape) > 3 else x[None, :]
        x = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = nn.elu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=16, kernel_size=(5, 5), padding="VALID")(x)
        x = nn.elu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(features=120)(x)
        x = nn.elu(x)
        x = nn.Dense(features=84)(x)
        x = nn.elu(x)
        x = nn.Dense(self.num_actions, name="last_layer")(x)
        return x.squeeze()


def setup(key: int):
    model = CNN(num_actions=10)
    key = jax.random.PRNGKey(key)
    key_env, key_env_step, key_act, key_init_params, key_run = jax.random.split(key, 5)
    params_init = model.init(key_init_params, jnp.ones((28, 28, 1)))

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("MNISTBandit-bsuite")

    # Reset the environment.
    context, state = env.reset(key_env, env_params)

    # Initialise agent
    params_init = model.init(key_init_params, context[..., None])

    # Sample a random action.
    action = env.action_space(env_params).sample(key_act)

    context_init, state_init, _, _, _ = env.step(key_env_step, state, action, env_params)

    env_stuff = (env, context_init, state_init, env_params)
    model_stuff = (model, params_init)

    return env_stuff, model_stuff, key_run



def step_ts(state, t, agent, env_params):
    key_step = jax.random.fold_in(key_run, t)
    key_step, key_take = jax.random.split(key_step)
    
    bel, context, env_state = state
    
    # Take action
    yhat = agent.sample_predictive(key_take, bel, context[..., None])
    action = yhat.argmax()
    
    # Obtain reward
    y = jax.nn.one_hot(env_state.correct_label, 10)[action]

    # Update belief
    X = (action, context[..., None])
    bel_update = agent.update(bel, y, X)

    # Take next step
    context_new, env_state, reward, done, _ = env.step(key_step, env_state, action, env_params)

    return (bel_update, context_new, env_state), (action, reward)


(env, context_init, state_init, env_params), (model, params_init), key_run = setup(314)

def mean_fn(params, x):
    if isinstance(x, tuple):
        action = x[0]
        x = x[1]
        eta = model.apply(params, x)
        mean = jax.nn.sigmoid(eta[action])
        return jnp.atleast_1d(mean)
    else:
        eta = model.apply(params, x)
        return eta


def cov_fn(eta, eps=0.1):
    mean = jax.nn.softmax(jnp.atleast_1d(eta))
    return jnp.diag(mean) - jnp.outer(mean, mean) + jnp.eye(len(mean)) * eps


def run_agent_ts(bel_init, context_init, state_init, env_params, n_steps):
    u_init = (bel_init, context_init, state_init)
    _step = partial(step_ts, agent=agent, env_params=env_params)
    (bel_final, *_), (actions, rewards) = jax.lax.scan(_step, u_init, jnp.arange(n_steps))
    rewards = jax.block_until_ready((rewards + 1) / 2)
    return actions, rewards


num_steps = 20_000

## Flores agent
print("Running Flores agent")
agent = flores.LowRankLastLayer(
    mean_fn,
    cov_fn,
    rank=50,
    dynamics_hidden=0.0,
    dynamics_last=0.0
)
bel_init = agent.init_bel(params_init, low_rank_diag=True, cov_hidden=1.0, cov_last=1.0)
actions, rewards = run_agent_ts(bel_init, context_init, state_init, env_params, num_steps)
print(rewards.sum())


## LoFi agent
print("Running LoFi agent")
dynamics = 1e-3
agent = lofi.LowRankPrecisionFilter(
    mean_fn, cov_fn, dynamics_covariance=dynamics, rank=50
)

bel_init = agent.init_bel(params_init, cov=1.0)
actions, rewards = run_agent_ts(bel_init, context_init, state_init, env_params, num_steps)
print(rewards.sum())