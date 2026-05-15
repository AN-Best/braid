import sys
import os
import sympy as sp
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index_reduction import order_reduction_pass
from sym_dae import SystemDAE

def test_order_reduction():
    dae = SystemDAE()
    t = dae.t
    
    x = sp.Function('x')(t)
    m = sp.Symbol('m')
    F = sp.Symbol('F')
    
    # Equation: m * x''(t) - F = 0
    eq = m * sp.Derivative(x, t, 2) - F
    
    dae.states.append(x)
    dae.params.extend([m, F])
    dae.equations.append(eq)
    
    print("Original Equations:")
    for eq in dae.equations:
        print(eq)
        
    new_dae = order_reduction_pass(dae)
    
    print("\nReduced Equations:")
    for eq in new_dae.equations:
        print(eq)
        
    print("\nStates:")
    print(new_dae.states)
    
    print("\nDerivatives Map:")
    print(new_dae.derivatives)

if __name__ == "__main__":
    test_order_reduction()