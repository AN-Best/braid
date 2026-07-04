"""
lowering/julia_lowering.py
===========================
Lowers a Braid IR dict to an allocation-free Julia ODE function string.
"""

def ast_to_julia_string(node: dict, state_names: list, param_names: list) -> str:
    """
    Recursively translates an AST node to a Julia code string.
    """
    op = node['op']

    if op == 'var':
        idx = state_names.index(node['name']) + 1
        return f"u[{idx}]"
    if op == 'param':
        idx = param_names.index(node['name']) + 1
        return f"p[{idx}]"
    if op == 'const':
        return str(float(node['value']))

    args = [ast_to_julia_string(a, state_names, param_names) for a in node['args']]

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
