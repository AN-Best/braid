"""
lowering/casadi_lowering.py
============================
Lowers a Braid IR dict back to a CasADi Function.

This is used by:
- backends/casadi_backend.py  (native SUNDIALS cvodes/idas)
- fmu_exporter.py             (C code generation via ca.Function.generate())

The round-trip is: CasADi SX → IR → CasADi Function (reconstruction).
The reconstructed Function is semantically equivalent to the original
but benefits from CasADi's CSE and JIT optimisations at codegen time.
"""

import casadi as ca


def _eval_ast(node: dict, x: ca.SX, p: ca.SX,
              state_idx: dict, param_idx: dict) -> ca.SX:
    """
    Recursively evaluate an IR AST node using CasADi SX.

    x: stacked state vector  (n_states × 1)
    p: stacked param vector  (n_params × 1)
    """
    op = node['op']

    if op == 'var':
        i = state_idx[node['name']]
        return x[i]
    if op == 'param':
        i = param_idx[node['name']]
        return p[i]
    if op == 'const':
        return ca.SX(float(node['value']))

    args = [_eval_ast(a, x, p, state_idx, param_idx) for a in node['args']]

    dispatch = {
        'add':  lambda a: a[0] + a[1],
        'sub':  lambda a: a[0] - a[1],
        'mul':  lambda a: a[0] * a[1],
        'div':  lambda a: a[0] / a[1],
        'neg':  lambda a: -a[0],
        'pow':  lambda a: a[0] ** a[1],
        'sin':  lambda a: ca.sin(a[0]),
        'cos':  lambda a: ca.cos(a[0]),
        'tan':  lambda a: ca.tan(a[0]),
        'exp':  lambda a: ca.exp(a[0]),
        'log':  lambda a: ca.log(a[0]),
        'sqrt': lambda a: ca.sqrt(a[0]),
        'abs':  lambda a: ca.fabs(a[0]),
        'min':  lambda a: ca.fmin(a[0], a[1]),
        'max':  lambda a: ca.fmax(a[0], a[1]),
        'ite':  lambda a: ca.if_else(a[0] != 0, a[1], a[2]),
    }

    if op not in dispatch:
        raise NotImplementedError(f"CasADi lowering: unsupported op '{op}'")

    return dispatch[op](args)


def make_casadi_function(ir: dict) -> ca.Function:
    """
    Reconstruct a CasADi Function from a Braid IR dict.

    Returns:
        ca.Function with signature  ode(x, p) → xdot
        where x is the stacked state vector and p is the stacked parameter vector.

    Example::

        fn = make_casadi_function(ir)
        xdot_val = fn(x_val, p_val)   # ca.DM inputs
    """
    n_states = len(ir['states'])
    n_params = len(ir['params'])

    x = ca.SX.sym('x', n_states)
    p = ca.SX.sym('p', n_params)

    state_idx = {name: i for i, name in enumerate(ir['states'])}
    param_idx = {name: i for i, name in enumerate(ir['params'])}
    rhs_asts  = [entry['expr'] for entry in ir['ode_rhs']]

    rhs_exprs = ca.vertcat(*[
        _eval_ast(ast, x, p, state_idx, param_idx)
        for ast in rhs_asts
    ])

    fn = ca.Function(
        'ode',
        [x, p],
        [rhs_exprs],
        ['x', 'p'],
        ['xdot'],
    )
    return fn


def make_casadi_jacobian(ir: dict) -> ca.Function:
    """
    Build the analytical Jacobian ∂xdot/∂x as a CasADi Function.

    Returns:
        ca.Function  jac(x, p) → J  where J has shape (n_states × n_states)
    """
    n_states = len(ir['states'])
    n_params = len(ir['params'])

    x = ca.SX.sym('x', n_states)
    p = ca.SX.sym('p', n_params)

    state_idx = {name: i for i, name in enumerate(ir['states'])}
    param_idx = {name: i for i, name in enumerate(ir['params'])}
    rhs_asts  = [entry['expr'] for entry in ir['ode_rhs']]

    rhs_exprs = ca.vertcat(*[
        _eval_ast(ast, x, p, state_idx, param_idx)
        for ast in rhs_asts
    ])

    J_sym = ca.jacobian(rhs_exprs, x)

    jac_fn = ca.Function(
        'jac',
        [x, p],
        [J_sym],
        ['x', 'p'],
        ['J'],
    )
    return jac_fn


def generate_c_code(ir: dict, output_dir: str = '.', func_name: str = 'braid_ode') -> str:
    """
    Generate optimised C code for the ODE from the IR using CasADi's codegen.

    CasADi applies Common Subexpression Elimination (CSE) automatically,
    producing significantly better C code than manual codegen.

    Args:
        ir:          Braid IR dict
        output_dir:  Directory to write the .c file
        func_name:   Name of the generated C function

    Returns:
        Path to the generated .c file.
    """
    import os
    fn = make_casadi_function(ir)

    gen = ca.CodeGenerator(func_name)
    gen.add(fn)

    os.makedirs(output_dir, exist_ok=True)
    gen.generate(output_dir + '/')
    return os.path.join(output_dir, f'{func_name}.c')
