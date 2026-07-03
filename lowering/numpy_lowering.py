"""
lowering/numpy_lowering.py
===========================
Lowers a Braid IR dict to a pure NumPy ODE function for use with SciPy.
"""

import numpy as np


def _eval_ast(node: dict, x: np.ndarray, p: np.ndarray,
              state_idx: dict, param_idx: dict):
    """
    Recursively evaluate an IR AST node using NumPy.

    x: state array  (n_states,)
    p: param array  (n_params,)
    """
    op = node['op']

    if op == 'var':
        return x[state_idx[node['name']]]
    if op == 'param':
        return p[param_idx[node['name']]]
    if op == 'const':
        return float(node['value'])

    args = [_eval_ast(a, x, p, state_idx, param_idx) for a in node['args']]

    dispatch = {
        'add':  lambda a: a[0] + a[1],
        'sub':  lambda a: a[0] - a[1],
        'mul':  lambda a: a[0] * a[1],
        'div':  lambda a: a[0] / a[1],
        'neg':  lambda a: -a[0],
        'pow':  lambda a: np.power(a[0], a[1]),
        'sin':  lambda a: np.sin(a[0]),
        'cos':  lambda a: np.cos(a[0]),
        'tan':  lambda a: np.tan(a[0]),
        'exp':  lambda a: np.exp(a[0]),
        'log':  lambda a: np.log(a[0]),
        'sqrt': lambda a: np.sqrt(a[0]),
        'abs':  lambda a: np.abs(a[0]),
        'min':  lambda a: np.minimum(a[0], a[1]),
        'max':  lambda a: np.maximum(a[0], a[1]),
        'ite':  lambda a: np.where(a[0] != 0.0, a[1], a[2]),
    }

    if op not in dispatch:
        raise NotImplementedError(f"NumPy lowering: unsupported op '{op}'")

    return dispatch[op](args)


def make_numpy_ode(ir: dict):
    """
    Returns a callable  f(t, x, p) → dxdt  using pure NumPy.

    Args:
        ir: Braid IR dict (from ir.from_json or ir.compile_to_ir)

    Returns:
        Callable with signature  f(t: float, x: np.ndarray, p: np.ndarray) → np.ndarray
        where x.shape == (n_states,) and p.shape == (n_params,).
    """
    state_idx = {name: i for i, name in enumerate(ir['states'])}
    param_idx = {name: i for i, name in enumerate(ir['params'])}
    rhs_asts  = [entry['expr'] for entry in ir['ode_rhs']]

    def ode_func(t: float, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.array([
            _eval_ast(ast, x, p, state_idx, param_idx)
            for ast in rhs_asts
        ], dtype=np.float64)

    return ode_func


def make_numpy_jacobian(ir: dict):
    """
    Returns a numerical Jacobian  J(t, x, p) → np.ndarray  via finite differences.

    For most simulations the analytical Jacobian from CasADi lowering is preferred,
    but this provides a pure-NumPy fallback.
    """
    ode_fn = make_numpy_ode(ir)
    n = len(ir['states'])
    eps = 1e-7

    def jac_func(t: float, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        J = np.zeros((n, n), dtype=np.float64)
        f0 = ode_fn(t, x, p)
        for j in range(n):
            xp = x.copy()
            xp[j] += eps
            J[:, j] = (ode_fn(t, xp, p) - f0) / eps
        return J

    return jac_func
