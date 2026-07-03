"""
ir.py
=====
Braid Intermediate Representation (IR).

Converts a compiled CasadiDAE (after index reduction + tearing) into a
portable, backend-agnostic JSON expression tree. This IR can then be
lowered to any target (PyTorch, NumPy, CasADi, etc.) without requiring
CasADi at runtime.

Expression tree node format
---------------------------
Every node is a dict with an 'op' field:

    Leaves:
        {"op": "var",   "name": "x_mass"}      — state variable
        {"op": "param", "name": "k_spring"}     — parameter
        {"op": "const", "value": 9.81}          — numeric literal

    Unary:
        {"op": "neg",  "args": [child]}
        {"op": "sin",  "args": [child]}
        {"op": "cos",  "args": [child]}
        {"op": "tan",  "args": [child]}
        {"op": "exp",  "args": [child]}
        {"op": "log",  "args": [child]}
        {"op": "sqrt", "args": [child]}
        {"op": "abs",  "args": [child]}

    Binary:
        {"op": "add",  "args": [left, right]}
        {"op": "sub",  "args": [left, right]}
        {"op": "mul",  "args": [left, right]}
        {"op": "div",  "args": [left, right]}
        {"op": "pow",  "args": [base, exp]}
        {"op": "min",  "args": [a, b]}
        {"op": "max",  "args": [a, b]}

    Ternary:
        {"op": "ite",  "args": [cond, then_, else_]}   — if-then-else

Supported CasADi ops are mapped via _OP_MAP below.
"""

import json
import casadi as ca
from casadi_dae import CasadiDAE


# ─────────────────────────────────────────────────────────────────────────────
# CasADi operator → IR op name
# ─────────────────────────────────────────────────────────────────────────────

_OP_MAP = {
    ca.OP_ADD:           'add',
    ca.OP_SUB:           'sub',
    ca.OP_MUL:           'mul',
    ca.OP_DIV:           'div',
    ca.OP_NEG:           'neg',
    ca.OP_SQRT:          'sqrt',
    ca.OP_EXP:           'exp',
    ca.OP_LOG:           'log',
    ca.OP_SIN:           'sin',
    ca.OP_COS:           'cos',
    ca.OP_TAN:           'tan',
    ca.OP_ABS:           'abs',
    ca.OP_POW:           'pow',
    ca.OP_FMIN:          'min',
    ca.OP_FMAX:          'max',
    ca.OP_IF_ELSE_ZERO:  'ite',
    ca.OP_SQ:            'sq',    # x² → handled as pow(x, 2)
    ca.OP_TWICE:         'twice', # 2*x → handled as mul(2, x)
    ca.OP_INV:           'inv',   # 1/x → handled as div(1, x)
}


# ─────────────────────────────────────────────────────────────────────────────
# Expression tree walker: ca.SX → AST dict
# ─────────────────────────────────────────────────────────────────────────────

def sx_to_ast(expr: ca.SX, state_names: set, param_names: set) -> dict:
    """
    Recursively convert a scalar CasADi SX expression into a JSON-serialisable
    AST dict.

    Args:
        expr:        Scalar CasADi SX expression.
        state_names: Set of state variable name strings (from dae.state_names).
        param_names: Set of parameter name strings (from dae.param_names).

    Returns:
        Nested dict representing the expression tree.
    """
    # ── Leaf: named symbolic variable ────────────────────────────────────────
    if expr.is_symbolic():
        name = expr.name()
        if name in state_names:
            return {'op': 'var',   'name': name}
        elif name in param_names:
            return {'op': 'param', 'name': name}
        else:
            # Unknown symbol (time variable, etc.) — treat as var
            return {'op': 'var', 'name': name}

    # ── Leaf: numeric constant ───────────────────────────────────────────────
    if expr.is_constant():
        value = float(expr)
        if value == int(value):
            value = int(value)
        return {'op': 'const', 'value': value}

    # ── Composite node ───────────────────────────────────────────────────────
    op_code = expr.op()
    op_name = _OP_MAP.get(op_code)

    if op_name is None:
        raise NotImplementedError(
            f"CasADi op code {op_code} is not supported in the Braid IR. "
            "Please file an issue if you need this operator."
        )

    n_dep = expr.n_dep()

    # ── Special-case normalizations ──────────────────────────────────────────

    # sq(x) → pow(x, 2)
    if op_name == 'sq':
        child = sx_to_ast(expr.dep(0), state_names, param_names)
        return {'op': 'pow', 'args': [child, {'op': 'const', 'value': 2}]}

    # twice(x) → mul(2, x)
    if op_name == 'twice':
        child = sx_to_ast(expr.dep(0), state_names, param_names)
        return {'op': 'mul', 'args': [{'op': 'const', 'value': 2}, child]}

    # inv(x) → div(1, x)
    if op_name == 'inv':
        child = sx_to_ast(expr.dep(0), state_names, param_names)
        return {'op': 'div', 'args': [{'op': 'const', 'value': 1}, child]}

    # ── General case ─────────────────────────────────────────────────────────
    args = [sx_to_ast(expr.dep(i), state_names, param_names) for i in range(n_dep)]
    return {'op': op_name, 'args': args}


# ─────────────────────────────────────────────────────────────────────────────
# Serialize: CasadiDAE → JSON string
# ─────────────────────────────────────────────────────────────────────────────

def to_json(dae: CasadiDAE) -> str:
    """
    Serialize a compiled CasadiDAE (post-tearing) to a Braid IR JSON string.

    Requires dae.ode_rhs to be populated (call tearing_pass first).

    Returns:
        JSON string — portable, no CasADi dependency to consume.
    """
    if not dae.ode_rhs:
        raise ValueError(
            "dae.ode_rhs is empty. Run tearing_pass() before serializing to JSON IR."
        )

    state_name_set = set(dae.state_names)
    param_name_set = set(dae.param_names)

    # Walk each ODE RHS expression and convert to AST
    ode_rhs_json = []
    for state_name, rhs_expr in zip(dae.state_names, dae.ode_rhs):
        try:
            ast = sx_to_ast(ca.SX(rhs_expr), state_name_set, param_name_set)
        except Exception as e:
            raise RuntimeError(
                f"Failed to serialize ODE RHS for state '{state_name}': {e}"
            )
        ode_rhs_json.append({
            'state': state_name,
            'expr':  ast,
        })

    data = {
        'version':    '2.0',
        'backend':    'braid-casadi',
        'states':     dae.state_names,
        'params':     dae.param_names,
        'param_meta': dae.param_meta,
        'components': dae.components,
        'ode_rhs':    ode_rhs_json,
        'sensor_mappings': dae.sensor_mappings,
    }

    return json.dumps(data, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Deserialize: JSON string / file → IR dict
# ─────────────────────────────────────────────────────────────────────────────

def from_json(s: str) -> dict:
    """
    Deserialize a Braid IR JSON string into a plain Python dict.

    The returned dict is consumed directly by lowering functions
    (make_torch_ode, make_numpy_ode, make_casadi_function, etc.).
    No CasADi dependency is needed to load or use the IR.

    Args:
        s: JSON string or file path (str ending with .json).

    Returns:
        Plain Python dict with keys: version, states, params,
        param_meta, components, ode_rhs, sensor_mappings.
    """
    import os
    if isinstance(s, (str, os.PathLike)):
        path_str = str(s)
        if os.path.isfile(path_str):
            with open(path_str, 'r', encoding='utf-8') as f:
                s = f.read()

    data = json.loads(s)

    if data.get('version') not in ('2.0',):
        raise ValueError(
            f"Unsupported Braid IR version '{data.get('version')}'. "
            "Expected '2.0'. Re-compile your model with the current Braid version."
        )

    return data


def save(dae: CasadiDAE, path: str) -> str:
    """Compile and save the IR to a .json file. Returns the absolute path."""
    import os
    ir_str = to_json(dae)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(ir_str)
    return os.path.abspath(path)


def load(path: str) -> dict:
    """Load a Braid IR from a .json file."""
    return from_json(path)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: full compile pipeline
# ─────────────────────────────────────────────────────────────────────────────

def compile_to_ir(system) -> dict:
    """
    Full pipeline convenience function:
        System  →  pantelides_pass  →  tearing_pass  →  IR dict

    Args:
        system: A braid.base.System instance with Node() connections applied.

    Returns:
        IR dict ready for lowering.
    """
    from index_reduction import pantelides_pass, tearing_pass

    dae = system.to_dae()
    dae = pantelides_pass(dae)
    dae = tearing_pass(dae)
    return json.loads(to_json(dae))
