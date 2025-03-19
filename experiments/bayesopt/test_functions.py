"""
Suite of 'stationary' benchmarks
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

class TrueMLP(nn.Module):
    n_hidden: int = 50

    def setup(self):
        self.hidden_1 = nn.Dense(50,
            bias_init=nn.initializers.normal(1.0),
            kernel_init=nn.initializers.normal(1.0),
        )

        self.hidden_2 = nn.Dense(50,
            bias_init=nn.initializers.normal(1.0),
            kernel_init=nn.initializers.normal(1.0),
        )

        self.last = nn.Dense(1,
            bias_init=nn.initializers.normal(1.0),
            kernel_init=nn.initializers.normal(1.0),
        )

    def __call__(self, x):
        x = self.hidden_1(x)
        x = nn.relu(x)
        x = self.hidden_2(x)
        x = nn.relu(x)
        x = self.last(x)
        return x
    

def init_fn_draw_nn(key, dim):
    X_init = jnp.ones((1, dim))
    base_model = TrueMLP()
    params_base = base_model.init(key, X_init)

    def objective_fn(x):
        return base_model.apply(params_base, x)

    return objective_fn


def ackley_1d(x, y=0):
    out = (-20*jnp.exp(-0.2*jnp.sqrt(0.5*(x**2 + y**2))) 
           - jnp.exp(0.5*(jnp.cos(2*jnp.pi*x) + jnp.cos(2*jnp.pi*y)))
           + jnp.e + 20)
    
    return out


def hartmann6(x):
    """
    Hartmann 6-Dimensional Function
    For reference, see https://www.sfu.ca/~ssurjano/hart6.html

    Evaluates the 6-dimensional Hartmann function at point x.
    :param x: A numpy array of shape (6,)
    :return: Function value at x
    """
    # Coefficients
    alpha = jnp.array([1.0, 1.2, 3.0, 3.2])
    A = jnp.array([
        [10, 3, 17, 3.50, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ])
    P = 1e-4 * jnp.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ])
    
    # Compute function value
    sum_terms = jnp.sum(A * (x - P) ** 2, axis=1)
    exp_terms = jnp.exp(-sum_terms)
    result = -jnp.sum(alpha * exp_terms)
    
    return result



def ackley(x, a=20, b=0.2, c=2*jnp.pi, lbound=-4, ubound=5):
    """
    Ackley function
    For reference, see https://www.sfu.ca/~ssurjano/ackley.html

    Evaluates the d-dimensional Ackley function at point x.
    :param x: A numpy array of shape (d,)
    :param a: Recommended value 20
    :param b: Recommended value 0.2
    :param c: Recommended value 2*pi
    :return: Function value at x
    """
    x = x * (ubound - lbound) + lbound
    d = len(x)
    sum1 = jnp.sum(x**2)
    sum2 = jnp.sum(jnp.cos(c * x))
    term1 = -a * jnp.exp(-b * jnp.sqrt(sum1 / d))
    term2 = -jnp.exp(sum2 / d)
    return term1 + term2 + a + jnp.exp(1)


def branin(x):
    """
    Branin function
    For reference, see https://www.sfu.ca/~ssurjano/branin.html

    Evaluates the Branin function in the rescaled domain [0,1]^2.
    :param x: A numpy array of shape (2,)
    :return: Function value at x
    """
    # Rescale x1 and x2 to the original domain
    x1 = 15 * x[0] - 5
    x2 = 15 * x[1]
    
    # Branin function parameters
    a = 1
    b = 5.1 / (4 * jnp.pi**2)
    c = 5 / jnp.pi
    r = 6
    s = 10
    t = 1 / (8 * jnp.pi)
    
    # Compute function value
    term1 = (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * jnp.cos(x1)
    result = term1 + term2 + s
    
    return result


EXPRIMENTS = {
    "hartmann": hartmann6,
    "branin": branin,
    "ackley2": ackley,
    "ackley5": ackley,
}
