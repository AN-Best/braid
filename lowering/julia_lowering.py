"""
lowering/julia_lowering.py
===========================
Lowers a Braid IR dict to an allocation-free Julia ODE function string.
"""

def ast_to_julia_string(node: dict, state_names: list, param_names: list, xdot_names: list = None) -> str:
    """
    Recursively translates an AST node to a Julia code string.
    """
    op = node['op']

    if op == 'var':
        name = node['name']
        if xdot_names is not None and name in xdot_names:
            idx = xdot_names.index(name) + 1
            return f"du[{idx}]"
        else:
            idx = state_names.index(name) + 1
            return f"u[{idx}]"
    if op == 'param':
        idx = param_names.index(node['name']) + 1
        return f"p[{idx}]"
    if op == 'const':
        return str(float(node['value']))

    args = [ast_to_julia_string(a, state_names, param_names, xdot_names) for a in node['args']]

    dispatch = {
        'add':  lambda a: f"({a[0]} + {a[1]})",
        'sub':  lambda a: f"({a[0]} - {a[1]})",
        'mul':  lambda a: f"({a[0]} * {a[1]})",
        'div':  lambda a: f"({a[0]} / {a[1]})",
        'neg':  lambda a: f"(-{a[0]})",
        'pow':  lambda a: f"({a[0]} ^ {a[1]})",
        'sin':  lambda a: f"sin({a[0]})",
        'cos':  lambda a: f"cos({a[0]})",
        'tan':  lambda a: f"tan({a[0]})",
        'exp':  lambda a: f"exp({a[0]})",
        'log':  lambda a: f"log({a[0]})",
        'sqrt': lambda a: f"sqrt({a[0]})",
        'abs':  lambda a: f"abs({a[0]})",
        'min':  lambda a: f"min({a[0]}, {a[1]})",
        'max':  lambda a: f"max({a[0]}, {a[1]})",
        'ite':  lambda a: f"({a[0]} != 0.0 ? {a[1]} : {a[2]})",
        'atan2': lambda a: f"atan({a[0]}, {a[1]})",
    }


    if op not in dispatch:
        raise NotImplementedError(f"Julia lowering: unsupported op '{op}'")

    return dispatch[op](args)


def generate_julia_ode_code(ir: dict, func_name: str = "f_ode") -> str:
    """
    Generates a complete, allocation-free Julia ODE function string:
        f(du, u, p, t)
    
    This function is compatible with GPU compilation (DiffEqGPU.jl / CUDA.jl).
    """
    state_names = ir['states']
    param_names = ir['params']
    
    lines = []
    lines.append(f"function {func_name}(du, u, p, t)")
    
    for i, entry in enumerate(ir['ode_rhs']):
        expr_str = ast_to_julia_string(entry['expr'], state_names, param_names)
        lines.append(f"    du[{i + 1}] = {expr_str}")
        
    lines.append("    nothing")
    lines.append("end")
    
    return "\n".join(lines)


def generate_julia_dae_code(ir: dict, func_name: str = "f_dae") -> str:
    """
    Generates a complete, allocation-free Julia DAE function string:
        f(out, du, u, p, t)
    where out[i] = residual_i = 0
    """
    state_names = ir['states']
    param_names = ir['params']
    xdot_names = ir.get('xdots', [])
    
    lines = []
    lines.append(f"function {func_name}(out, du, u, p, t)")
    
    for i, entry in enumerate(ir['residuals']):
        expr_str = ast_to_julia_string(entry, state_names, param_names, xdot_names)
        lines.append(f"    out[{i + 1}] = {expr_str}")

        
    lines.append("    nothing")
    lines.append("end")
    
    return "\n".join(lines)

