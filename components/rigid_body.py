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
