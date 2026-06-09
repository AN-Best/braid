import json
import sympy as sp
from sym_dae import SystemDAE

def to_json(dae: SystemDAE) -> str:
    """Serialize a SystemDAE object to a JSON string.
    Uses SymPy's srepr to represent all symbolic expressions.
    """
    # Determine the form of the reduced system (ODE or DAE) and index
    solved_assignments = getattr(dae, "solved_assignments", {})
    if solved_assignments:
        system_form = "ODE"
        index = 0
    else:
        is_dae = any(isinstance(v, sp.Function) for v in dae.solved_variables)
        system_form = "DAE" if is_dae else "ODE"
        if is_dae:
            d_indices = getattr(dae, "differentiation_indices", [])
            index = max(d_indices) + 1 if d_indices else 1
        else:
            index = 0

    data = {
        "t": sp.srepr(dae.t),
        "params": [sp.srepr(p) for p in dae.params],
        "states": [sp.srepr(s) for s in dae.states],
        "solved_variables": [sp.srepr(v) for v in dae.solved_variables],
        "active_equations": [sp.srepr(eq) for eq in dae.active_equations],
        "equations": [sp.srepr(eq) for eq in dae.equations],
        "matching": {str(k): sp.srepr(v) for k, v in dae.matching.items()},
        "differentiation_indices": getattr(dae, "differentiation_indices", []),
        "components": getattr(dae, "components", []),
        "param_meta": getattr(dae, "param_meta", {}),
        "derivatives": {sp.srepr(k): sp.srepr(v) for k, v in dae.derivatives.items()},
        "solved_assignments": {sp.srepr(k): sp.srepr(v) for k, v in solved_assignments.items()},
        "ode_assignments": {sp.srepr(k): sp.srepr(v) for k, v in getattr(dae, "ode_assignments", {}).items()},
        "system_form": system_form,
        "index": index
    }
    return json.dumps(data, indent=2)

def from_json(json_str: str) -> SystemDAE:
    """Deserialize a JSON string back into a SystemDAE object.
    Uses SymPy's sympify to parse srepr expressions.
    """
    data = json.loads(json_str)
    dae = SystemDAE()
    
    dae.t = sp.sympify(data["t"])
    dae.params = [sp.sympify(p) for p in data["params"]]
    dae.states = [sp.sympify(s) for s in data["states"]]
    dae.solved_variables = [sp.sympify(v) for v in data["solved_variables"]]
    dae.active_equations = [sp.sympify(eq) for eq in data["active_equations"]]
    dae.equations = [sp.sympify(eq) for eq in data["equations"]]
    dae.matching = {int(k): sp.sympify(v) for k, v in data["matching"].items()}
    dae.differentiation_indices = data["differentiation_indices"]
    dae.components = data["components"]
    dae.param_meta = data.get("param_meta", {})
    dae.derivatives = {sp.sympify(k): sp.sympify(v) for k, v in data["derivatives"].items()}
    dae.solved_assignments = {sp.sympify(k): sp.sympify(v) for k, v in data.get("solved_assignments", {}).items()}
    dae.ode_assignments = {sp.sympify(k): sp.sympify(v) for k, v in data.get("ode_assignments", {}).items()}
    dae.system_form = data.get("system_form", "DAE")
    dae.index = data.get("index", 0)
    
    return dae
