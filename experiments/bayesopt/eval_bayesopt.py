import jax
import numpy as np
import jax.numpy as jnp
from time import time
from bayesopt import test_functions, eval_fn, agents

def objective_fn(x):
    return -test_functions.branin(x)

x_test = jnp.zeros(2)
dim = len(x_test)

# def objective_fn(x):
#     return -test_functions.hartmann6(x)
# x_test = jnp.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
# dim = len(x_test)

# def objective_fn(x):
#     return -test_functions.ackley(x, lbound=-4, ubound=5)
# x_test = jnp.zeros(5)
# dim = len(x_test)

n_runs = 10

agent_list = []
bel_fn_list = []
times_list = []
names = ["GP", "LRKF", "FLoRES", "LoFi"]


agent, bel_init_fn = agents.load_gp_agent(
    x_test,
    lenght_scale=0.1 + jnp.sqrt(dim),
    nu=5/2,
    buffer_size=50,
    obs_noise=0.0,
)

agent_list.append(agent)
bel_fn_list.append(bel_init_fn)

agent, bel_init_fn = agents.load_lrkf_agent(
    x_test, rank=50, cov=1.0, obs_noise=0.0, dynamics_cov=0.0, low_rank_diag=False,
)

agent_list.append(agent)
bel_fn_list.append(bel_init_fn)

agent, bel_init_fn = agents.load_ll_lrkf_agent(
    x_test, rank=100,
    cov_hidden=1e-4, # parameters do not vary much from their initial parameters
    cov_last=0.1, # uncertainty in target
    low_rank_diag=False,
)

agent_list.append(agent)
bel_fn_list.append(bel_init_fn)

agent, bel_init_fn = agents.load_lofi_agent(
    x_test, rank=50, cov_init=1e-2, obs_noise=0.01, dynamics_covariance=1e-5
)

agent_list.append(agent)
bel_fn_list.append(bel_init_fn)

if __name__ == "__main__":
    key = jax.random.PRNGKey(314)
    n_steps = 100
    lbound, ubound = 0.0, 1.0 # for hartman and branin
    keys = jax.random.split(key, n_runs)
    res = {}
    for name, agent, init_fn in zip(names, agent_list, bel_fn_list):
        print(f"Eval {name}")
        time_init = time()
        runs = eval_fn.test_runs(
            keys, n_steps, agent, init_fn, objective_fn, dim, lbound, ubound, dim
        )
        runs = jax.tree.map(np.array, runs)
        time_end = time()
        res[name] = {**runs, "time": time_end - time_init}
