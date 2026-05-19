import sympy as sp
from sym_dae import SystemDAE

def order_reduction_pass(dae: SystemDAE) -> SystemDAE:
    new_dae = dae.copy_structure()
    t = new_dae.t
    
    for eq in dae.equations:
        new_eq = eq
        
        # Find all derivatives in the equation
        derivatives = eq.find(sp.Derivative)
        
        for deriv in derivatives:
            # Check if it's a second derivative with respect to time t
            if len(deriv.variables) == 2 and all(v == t for v in deriv.variables):
                # The state function being differentiated, e.g., x(t)
                state_func = deriv.expr
                
                # Create a new state function for the first derivative, e.g., x_dot(t)
                state_name = state_func.func.__name__
                v_name = f"{state_name}_dot"
                v_func = sp.Function(v_name)(t)
                
                # Add to states if not already there
                if v_func not in new_dae.states:
                    new_dae.states.append(v_func)
                    
                    # Store derivative mapping
                    new_dae.derivatives[state_func] = v_func
                    
                    # Add definition equation: x'(t) - x_dot(t) = 0
                    def_eq = sp.Derivative(state_func, t) - v_func
                    new_dae.equations.append(def_eq)
                
                # Replace x''(t) with v'(t) in the current equation
                new_deriv = sp.Derivative(v_func, t)
                new_eq = new_eq.replace(deriv, new_deriv)
                
        new_dae.equations.append(new_eq)
        
    return new_dae