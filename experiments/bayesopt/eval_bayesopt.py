import os
import jax
import sys
import toml
import pickle
import numpy as np
import jax.numpy as jnp
from time import time
from bayesopt import test_functions, eval_fn, agents

_, base_path, fn_string = sys.argv
path_dataset = os.path.join(base_path, "datasets.toml")
path_agents = os.path.join(base_path, "agents.toml")

with open(path_agents, "r") as f:
    config_agents = toml.load(f)

with open(path_dataset, "r") as f:
    config_experiment = toml.load(f)
    dim = config_experiment["experiment"][fn_string]["dim"]
    n_runs = config_experiment["config"]["n_runs"]
    exploration_method = config_experiment["config"]["exploration_method"]
    key = int(config_experiment["config"]["key"])
    n_steps = config_experiment["experiment"][fn_string]["n_eval"]
    query_method = config_experiment["experiment"][fn_string]["query_method"]
    x_test = jnp.zeros(dim)

def objective_fn(x):
    return -test_functions.EXPRIMENTS[fn_string](x)

key = jax.random.PRNGKey(key)
lbound, ubound = 0.0, 1.0
keys = jax.random.split(key, n_runs)

res = {}
print(f"*** Running {fn_string} ***")
for name, load_agent in agents.AGENTS.items():
    if (name == "GP") and (query_method == "grad"):
        print("GP agent not available for grad query method")
        continue

    print(f"Eval {name}")
    config_agent = config_agents[name]
    agent, init_fn = load_agent(x_test, **config_agent)
    time_init = time()
    runs = eval_fn.test_runs(
        keys, n_steps, agent, init_fn, objective_fn, dim, lbound, ubound, dim, query_method, exploration_method
    )
    runs = jax.tree.map(np.array, runs)
    time_end = time()
    res[name] = {**runs, "time": time_end - time_init}

with open(f"{fn_string}_{exploration_method}.pkl", "wb") as f:
    pickle.dump(res, f)
