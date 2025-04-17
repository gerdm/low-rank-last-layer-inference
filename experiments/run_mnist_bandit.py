import os
import jax
import click
import pickle
import gymnax
import numpy as np
import jax.numpy as jnp
from time import time
from functools import partial
from agents_mnist import agents, model


def setup_env(key):
    key_env, key_env_step, key_act, key_run = jax.random.split(key, 4)

    # Instantiate the environment & its settings.
    env, env_params = gymnax.make("MNISTBandit-bsuite")

    # Reset the environment.
    context, state = env.reset(key_env, env_params)

    # Sample a random action.
    action = env.action_space(env_params).sample(key_act)

    context_init, state_init, _, _, _ = env.step(key_env_step, state, action, env_params)

    env_stuff = (env, context_init, state_init, env_params)

    return env_stuff, key_run


def step_egreedy(state, t, agent, key_base, env, env_params, eps):
    """
    Step for epsilon-greedy agent
    """
    key_step = jax.random.fold_in(key_base, t)
    key_step, key_take = jax.random.split(key_step)
    
    bel, context, env_state = state

    # Take action
    key_eps, key_take = jax.random.split(key_take)
    yhat = agent.predict_fn(bel, context.ravel())
    action = yhat.argmax()
    take_random = jax.random.bernoulli(key_eps, p=eps)
    action_random = jax.random.choice(key_take, 10)

    action = action * (1 - take_random) + action_random * take_random
    action = action.astype(int)
    
    # Obtain reward
    y = jax.nn.one_hot(env_state.correct_label, 10)[action]

    # Update belief
    # X = (action, context[..., None])
    X = jnp.concat([jnp.array([action]), context.ravel()])
    bel_update = agent.update(bel, y, X)
    
    # Take next step
    context_new, env_state, reward, done, _ = env.step(key_step, env_state, action, env_params)

    return (bel_update, context_new, env_state), (action, reward)


def step_ts(state, t, agent, key_base, env, env_params):
    key_step = jax.random.fold_in(key_base, t)
    key_step, key_take = jax.random.split(key_step)
    
    bel, context, env_state = state
    
    # Take action
    yhat = agent.sample_predictive(key_take, bel, context.ravel())
    action = yhat.argmax()
    
    # Obtain reward
    y = jax.nn.one_hot(env_state.correct_label, 10)[action]

    # Update belief
    X = jnp.concat([jnp.array([action]), context.ravel()])
    bel_update = agent.update(bel, y, X)

    # Take next step
    context_new, env_state, reward, done, _ = env.step(key_step, env_state, action, env_params)

    return (bel_update, context_new, env_state), (action, reward)


def run_agent(
    key, agent, bel_init, n_steps, step_fn, step_fn_kwargs
):
    (env, context_init, state_init, env_params), key_run = setup_env(key)

    u_init = (bel_init, context_init, state_init)
    _step = partial(step_fn, agent=agent, key_base=key_run, env=env, env_params=env_params, **step_fn_kwargs)
    (bel_final, *_), (actions, rewards) = jax.lax.scan(_step, u_init, jnp.arange(n_steps))
    rewards = jax.block_until_ready((rewards + 1) / 2)
    return bel_final, actions, rewards

@partial(jax.vmap, in_axes=(0, None, 0, None, None, None))
def run_agents(
    key, agent, bel_init, n_steps, step_fn, step_fn_kwargs
):
    bel_final, action, rewards = run_agent(key, agent, bel_init, n_steps, step_fn, step_fn_kwargs)
    return action, rewards


def optimize_agent(
    key, agent, bel_init, n_steps, step_fn, step_fn_kwargs
):
    ...


@click.command()
@click.option("--agent", help="Agent to run")
@click.option("--key", default=314, help="Random key")
@click.option("--eps", default=0.05, help="Epsilon value")
@click.option("--num_steps", default=40_000, help="Total number of observations")
@click.option("--base_path", default=".", help="Base path for saving results")
@click.option("--num_trials", default=1, help="Number of simulations ot run")
def run_epsilon_greedy(agent, key, eps, num_steps, base_path, num_trials):
    print(f"Running {agent} agent")
    key = jax.random.PRNGKey(key)
    key_params, key_run = jax.random.split(key)

    keys_params = jax.random.split(key_params, num_trials)
    params_init_runs = jax.vmap(model.init, in_axes=(0, None))(keys_params, jnp.ones((28, 28, 1)))

    agent_instance, init_kwargs = agents[agent]()

    @jax.vmap
    def init_agent(params):
        bel_init = agent_instance.init_bel(params, **init_kwargs)
        return bel_init

    bel_init_runs = init_agent(params_init_runs)

    step_fn_config = {"eps": eps}
    keys_run = jax.random.split(key_run, num_trials)
    time_init = time()
    actions, rewards = run_agents(keys_run, agent_instance, bel_init_runs, num_steps, step_egreedy, step_fn_config)
    res = {
        "actions": actions,
        "rewards": rewards,
        "eps": eps,
    }
    res = jax.tree.map(np.array, res)
    time_end = time()
    res = {"time": time_end - time_init, **res}

    filename = f"{agent}_eps_{eps * 100:.0f}.pkl"
    path_out = os.path.join(base_path, filename)
    with open(path_out, "wb") as f:
        pickle.dump(res, f)
    average_reward = np.mean(rewards.sum(axis=-1), axis=0)
    print(f"Results saved to {path_out}")
    print(f"Average cumulative reward: {average_reward:0.2f}")
    print(f"Total time: {time_end - time_init:.4f} seconds", end="\n\n")


@click.command()
@click.option("--agent", help="Agent to run")
@click.option("--key", default=314, help="Random key")
@click.option("--num_steps", default=40_000, help="Total number of observations")
@click.option("--base_path", default=".", help="Base path for saving results")
@click.option("--num_trials", default=1, help="Number of simulations ot run")
def run_ts(agent, key, num_steps, base_path, num_trials):
    print(f"Running {agent} agent")
    key = jax.random.PRNGKey(key)
    key_params, key_run = jax.random.split(key)

    keys_params = jax.random.split(key_params, num_trials)
    params_init_runs = jax.vmap(model.init, in_axes=(0, None))(keys_params, jnp.ones((28, 28, 1)))

    agent_instance, init_kwargs = agents[agent]()

    @jax.vmap
    def init_agent(params):
        bel_init = agent_instance.init_bel(params, **init_kwargs)
        return bel_init

    bel_init_runs = init_agent(params_init_runs)

    step_fn_config = {}
    keys_run = jax.random.split(key_run, num_trials)
    time_init = time()
    actions, rewards = run_agents(keys_run, agent_instance, bel_init_runs, num_steps, step_ts, step_fn_config)

    res = {
        "actions": actions,
        "rewards": rewards,
    }
    res = jax.tree.map(np.array, res)
    time_end = time()
    res = {"time": time_end - time_init, **res}

    filename = f"{agent}_ts.pkl"
    path_out = os.path.join(base_path, filename)
    with open(path_out, "wb") as f:
        pickle.dump(res, f)
    
    average_reward = np.mean(rewards.sum(axis=-1), axis=0)
    print(f"Results saved to {path_out}")
    print(f"Average cumulative reward: {average_reward:0.2f}")
    print(f"Total time: {time_end - time_init:.4f} seconds", end="\n\n")


@click.group()
def cli():
    pass


cli.add_command(run_epsilon_greedy)
cli.add_command(run_ts)


if __name__ == "__main__":
    """
    Example usage:
    python -W ignore run_mnist_bandit.py run_epsilon_greedy --base_path output --num_trials 10 --agent OGD-adamw
    python -W ignore run_mnist_bandit.py run_ts --base_path output --num_trials 10 --agent OGD-adamw
    """
    cli()
