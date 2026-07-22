import casadi as ca
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from base import Component

class Generator(Component):
    """
    Ideal Electromechanical Generator component.
    
    Coupled equations:
      1. Back-EMF: V_p - V_n = ke * omega
      2. Torque:   tau = - kt * I_p
      3. KCL:      I_p + I_n = 0
    """
    def __init__(self, name: str, kt: float = 2.0, ke: float = 2.0):
        super().__init__(name)
        
        # 1. Register parameters
        kt_sym = ca.SX.sym(f"kt_{name}")
        ke_sym = ca.SX.sym(f"ke_{name}")
        self.register_param("kt", kt_sym, default=kt)
        self.register_param("ke", ke_sym, default=ke)
        
        # 2. Define internal variables
        tau_m = ca.SX.sym(f"tau_m_{name}")
        omega_m = ca.SX.sym(f"omega_m_{name}")
        
        v_p = ca.SX.sym(f"v_p_{name}")
        v_n = ca.SX.sym(f"v_n_{name}")
        i_p = ca.SX.sym(f"i_p_{name}")
        i_n = ca.SX.sym(f"i_n_{name}")
        v_p_dot = ca.SX.sym(f"v_p_{name}_dot")
        v_n_dot = ca.SX.sym(f"v_n_{name}_dot")
        
        # 3. Add equations
        self.equations = [
            # KCL
            i_p + i_n,
            # Back-EMF
            (v_p - v_n) - ke_sym * omega_m,
            # Torque opposing rotation
            tau_m + kt_sym * i_p
        ]
        
        # 4. Register ports
        # Mechanical shaft connection
        self.ports["shaft"] = [tau_m, ca.SX(0), omega_m]
        # Electrical terminals
        self.ports["p"] = [i_p, v_p, v_p_dot]
        self.ports["n"] = [i_n, v_n, v_n_dot]
