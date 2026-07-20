"""
components/rigid_body.py
========================
Rigid body dynamics components using urdf2casadi.
"""

import casadi as ca
from base import Component
from urdf2casadi.urdfparser import URDFparser

class RigidBodyURDF(Component):
    """
    Multibody rigid body dynamics component loaded from a URDF file.
    Uses urdf2casadi to generate forward dynamics via the Articulated Body Algorithm (ABA).
    Supports external spatial forces applied to designated links using Jacobian mapping.
    """
    def __init__(self, name: str, urdf_path: str, root_link: str, tip_link: str, gravity=None, external_force_links=None):
        super().__init__(name)
        
        self.urdf_path = urdf_path
        self.root_link = root_link
        self.tip_link = tip_link
        self.external_force_links = external_force_links or []
        
        # Parse URDF
        self.parser = URDFparser()
        self.parser.func_opts = {}
        self.parser.from_file(urdf_path)
        
        # Get joint information
        joint_info = self.parser.get_joint_info(root_link, tip_link)
        self.joint_names = joint_info[1]  # actuated joint names
        self.n_joints = len(self.joint_names)
        
        # Define states for each joint: position (q) and velocity (v)
        self.q_states = []
        self.qdot_states = []
        self.v_states = []
        self.vdot_states = []
        
        for jname in self.joint_names:
            q_s, qdot_s = self.add_state(f"q_{self.name}_{jname}")
            v_s, vdot_s = self.add_state(f"v_{self.name}_{jname}")
            self.q_states.append(q_s)
            self.qdot_states.append(qdot_s)
            self.v_states.append(v_s)
            self.vdot_states.append(vdot_s)
            
        self.q_vec = ca.vertcat(*self.q_states)
        self.qdot_vec = ca.vertcat(*self.qdot_states)
        self.v_vec = ca.vertcat(*self.v_states)
        self.vdot_vec = ca.vertcat(*self.vdot_states)
        
        # Inputs: control torques for each joint
        self.tau_syms = [ca.SX.sym(f"tau_{self.name}_{jname}") for jname in self.joint_names]
        self.tau_control_vec = ca.vertcat(*self.tau_syms)
        
        # Calculate Jacobians and equivalent joint torques for external force ports
        self.tau_ext_total = ca.SX.zeros(self.n_joints)
        
        # Ports dictionary: {port_name: [effort, across, dacross_dt]}
        self.ports = {}
        
        # Expose joint ports (torque control inputs)
        for i, jname in enumerate(self.joint_names):
            self.ports[jname] = [self.tau_syms[i], self.q_states[i], self.qdot_states[i]]
            
        # Add external force ports for requested links
        for link_name in self.external_force_links:
            # 6D spatial force input symbol: [Mx, My, Mz, Fx, Fy, Fz]
            f_ext_sym = ca.SX.sym(f"f_ext_{self.name}_{link_name}", 6)
            
            # Forward kinematics for this link: T_fk is 4x4 homogenous matrix function
            fk_dict = self.parser.get_forward_kinematics(root_link, link_name)
            T_fk_func = fk_dict["T_fk"]
            chain_joint_names = fk_dict["joint_names"]
            
            # Map chain joint names to component joint indices
            chain_indices = [self.joint_names.index(name) for name in chain_joint_names]
            
            # Sliced state vectors for the chain
            q_chain = ca.vertcat(*[self.q_states[idx] for idx in chain_indices]) if chain_indices else ca.SX(0, 1)
            v_chain = ca.vertcat(*[self.v_states[idx] for idx in chain_indices]) if chain_indices else ca.SX(0, 1)
            
            # Evaluate forward kinematics with the sliced joint state
            T_val = T_fk_func(q_chain)
            
            # Linear position is the 3D translation vector (top right 3x1 of T_fk)
            pos_link = T_val[0:3, 3]
            
            # Compute the linear velocity Jacobian: J_linear = d(pos_link)/dq (relative to component q_vec)
            J_linear = ca.jacobian(pos_link, self.q_vec)
            
            # To get rotation Jacobian: we can extract rotation matrix R = T[0:3, 0:3]
            # In spatial algebra, mapping of angular velocity is: omega = J_angular * qdot
            # Since urdf2casadi uses quaternions internally, we can compute it using quaternion derivative
            # For simplicity in translational mechanical acausal loading (e.g., ground reaction force / wind force):
            # We map the 3D linear force [Fx, Fy, Fz] to equivalent joint torques:
            # tau_ext = J_linear^T * f_linear
            f_linear = f_ext_sym[3:6]
            tau_ext = ca.mtimes(J_linear.T, f_linear)
            self.tau_ext_total = self.tau_ext_total + tau_ext
            
            # Expose as a 3D force port: [effort=f_linear, across=pos_link, dacross_dt=J_linear * v]
            v_link = ca.mtimes(J_linear, self.v_vec)
            self.ports[f"f_ext_{link_name}"] = [f_linear, pos_link, v_link]
            
        # Net joint torques including control torques and mapped external forces
        self.tau_net_vec = self.tau_control_vec + self.tau_ext_total
        
        # Get forward dynamics function from urdf2casadi without passing external force (avoiding upstream bug)
        if gravity is None:
            gravity = [0.0, 0.0, -9.81]
        
        aba_func = self.parser.get_forward_dynamics_aba(root_link, tip_link, gravity=gravity)
        vdot_expr = aba_func(self.q_vec, self.v_vec, self.tau_net_vec)
        
        # Setup Braid equations
        # 1. Kinematic relationship: dq/dt = v
        for i in range(self.n_joints):
            self.equations.append(self.qdot_states[i] - self.v_states[i])
            
        # 2. Dynamic relationship: dv/dt = ABA(q, v, tau_net)
        for i in range(self.n_joints):
            self.equations.append(self.vdot_states[i] - vdot_expr[i])


def sympy_to_casadi(expr, symbol_map):
    """
    Recursively converts a SymPy expression to a CasADi SX expression.
    symbol_map: dict mapping SymPy expressions to CasADi expressions.
    """
    import sympy as sp
    import casadi as ca

    if expr in symbol_map:
        return symbol_map[expr]

    if isinstance(expr, (sp.Integer, sp.Float)):
        return float(expr)

    if isinstance(expr, sp.Rational):
        return float(expr.p) / float(expr.q)

    if expr == sp.pi:
        return ca.pi

    if expr.is_Symbol:
        if expr.name in symbol_map:
            return symbol_map[expr.name]
        return ca.SX.sym(expr.name)

    if isinstance(expr, sp.core.function.AppliedUndef):
        if str(expr) in symbol_map:
            return symbol_map[str(expr)]
        return ca.SX.sym(str(expr.func))

    if isinstance(expr, sp.Eq):
        return sympy_to_casadi(expr.lhs - expr.rhs, symbol_map)

    # Recursively convert args
    args = [sympy_to_casadi(arg, symbol_map) for arg in expr.args]

    if isinstance(expr, sp.Add):
        return sum(args)
    elif isinstance(expr, sp.Mul):
        res = args[0]
        for arg in args[1:]:
            res = res * arg
        return res
    elif isinstance(expr, sp.Pow):
        return args[0]**args[1]
    elif isinstance(expr, sp.sin):
        return ca.sin(args[0])
    elif isinstance(expr, sp.cos):
        return ca.cos(args[0])
    elif isinstance(expr, sp.tan):
        return ca.tan(args[0])
    elif isinstance(expr, sp.asin):
        return ca.asin(args[0])
    elif isinstance(expr, sp.acos):
        return ca.acos(args[0])
    elif isinstance(expr, sp.atan):
        return ca.atan(args[0])
    elif isinstance(expr, sp.atan2):
        return ca.atan2(args[0], args[1])
    elif isinstance(expr, sp.exp):
        return ca.exp(args[0])
    elif isinstance(expr, sp.log):
        return ca.log(args[0])
    elif isinstance(expr, sp.sqrt):
        return ca.sqrt(args[0])
    else:
        raise TypeError(f"Unsupported SymPy operator / function: {type(expr)}: {expr}")


class RigidBodySymPy(Component):
    """
    Multibody rigid body dynamics component loaded from SymPy equations of motion.
    Converts SymPy physics/mechanics models (e.g. KanesMethod, LagrangesMethod, or explicit symbolic equations)
    to CasADi SX expressions.
    """
    def __init__(self, name: str, q, u=None, equations=None, control_symbols=None, param_symbols=None, ports=None):
        super().__init__(name)
        
        import sympy as sp
        
        control_symbols = control_symbols or []
        param_symbols = param_symbols or {}
        ports = ports or {}
        
        t = sp.Symbol('t')
        
        if u is None:
            u = [sp.Derivative(qi, t) for qi in q]
            
        self.q_sympy = list(q)
        self.u_sympy = list(u)
        
        self.q_states = []
        self.qdot_states = []
        self.u_states = []
        self.udot_states = []
        
        symbol_map = {}
        
        # Add states for coordinate variables q
        for qi in self.q_sympy:
            q_name = str(qi)
            if '(' in q_name:
                q_name = q_name.split('(')[0]
            q_s, qdot_s = self.add_state(f"q_{self.name}_{q_name}")
            self.q_states.append(q_s)
            self.qdot_states.append(qdot_s)
            
            symbol_map[qi] = q_s
            symbol_map[sp.Derivative(qi, t)] = qdot_s
            
        # Add states for speed variables u
        for ui in self.u_sympy:
            if isinstance(ui, sp.Derivative):
                inner_name = str(ui.args[0])
                if '(' in inner_name:
                    inner_name = inner_name.split('(')[0]
                u_name = f"{inner_name}_dot"
            else:
                u_name = str(ui)
                if '(' in u_name:
                    u_name = u_name.split('(')[0]
            u_s, udot_s = self.add_state(f"u_{self.name}_{u_name}")
            self.u_states.append(u_s)
            self.udot_states.append(udot_s)
            
            symbol_map[ui] = u_s
            symbol_map[sp.Derivative(ui, t)] = udot_s
            
        # Register parameters
        self.param_syms = {}
        for p_sym, default_val in param_symbols.items():
            p_ca = ca.SX.sym(p_sym.name)
            self.register_param(p_sym.name, p_ca, default=default_val)
            symbol_map[p_sym] = p_ca
            self.param_syms[p_sym] = p_ca
            
        # Register control inputs
        self.control_syms = {}
        for c_sym in control_symbols:
            c_ca = ca.SX.sym(c_sym.name)
            symbol_map[c_sym] = c_ca
            self.control_syms[c_sym] = c_ca
            
        # Convert equations
        from sympy.physics.mechanics import KanesMethod, LagrangesMethod
        
        extracted_eqs = []
        if isinstance(equations, KanesMethod):
            kindiff = equations.kindiffdict()
            for qi_dot, expr in kindiff.items():
                extracted_eqs.append(qi_dot - expr)
                
            M = equations.mass_matrix
            f = equations.forcing
            udot_vec = sp.Matrix([sp.Derivative(ui, t) for ui in self.u_sympy])
            dyn_eqs = M * udot_vec - f
            for eq in dyn_eqs:
                extracted_eqs.append(eq)
                
        elif isinstance(equations, LagrangesMethod):
            M = equations.mass_matrix
            f = equations.forcing
            qdot_vec = sp.Matrix([sp.Derivative(qi, t) for qi in self.q_sympy])
            qddot_vec = sp.Matrix([sp.Derivative(q_dot, t) for q_dot in qdot_vec])
            dyn_eqs = M * qddot_vec - f
            for eq in dyn_eqs:
                extracted_eqs.append(eq)
                
        elif isinstance(equations, (list, tuple)):
            extracted_eqs = list(equations)
        elif equations is not None:
            extracted_eqs = [equations]
            
        for eq in extracted_eqs:
            ca_eq = sympy_to_casadi(eq, symbol_map)
            self.equations.append(ca_eq)
            
        # If we don't have kinematic equations, add qdot_s - u_s = 0
        if not isinstance(equations, KanesMethod):
            if len(self.equations) <= len(self.u_states):
                for i in range(len(self.q_states)):
                    self.equations.insert(i, self.qdot_states[i] - self.u_states[i])

        # Convert ports
        for p_name, (eff_sp, acr_sp, dacr_sp) in ports.items():
            eff_ca = sympy_to_casadi(eff_sp, symbol_map)
            acr_ca = sympy_to_casadi(acr_sp, symbol_map)
            dacr_ca = sympy_to_casadi(dacr_sp, symbol_map)
            self.ports[p_name] = [eff_ca, acr_ca, dacr_ca]
            
        # Expose control symbols as ports
        for c_sym, c_ca in self.control_syms.items():
            if c_sym.name not in self.ports:
                self.ports[c_sym.name] = [c_ca, ca.SX(0), ca.SX(0)]

