import sympy as sp
from base import Component

t = sp.Symbol('t')

def skew(w):
    return sp.Matrix([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])

class World(Component):
    def __init__(self, name):
        super().__init__(name)

        r_sym = sp.Matrix([0, 0, 0])
        R_sym = sp.eye(3)
        v_sym = sp.Matrix([0, 0, 0])
        w_sym = sp.Matrix([0, 0, 0])

        f_sym = sp.Matrix([sp.Function(f'f_{self.name}_{i}')(t) for i in range(3)])
        tau_sym = sp.Matrix([sp.Function(f'tau_{self.name}_{i}')(t) for i in range(3)])

        # World has no differential states, it's a fixed reference
        self.equations = []

        self.ports = {'p': [[f_sym, tau_sym], [r_sym, R_sym], [v_sym, w_sym]]}


class RigidBody(Component):
    def __init__(self, name, m=1.0, Ixx=1.0, Iyy=1.0, Izz=1.0, g=[0.0, 0.0, -9.81]):
        super().__init__(name)

        # Parameters
        m_sym = sp.Symbol(f'm_{self.name}')
        self.register_param('m', m_sym, default=m)

        Ixx_sym = sp.Symbol(f'Ixx_{self.name}')
        Iyy_sym = sp.Symbol(f'Iyy_{self.name}')
        Izz_sym = sp.Symbol(f'Izz_{self.name}')
        self.register_param('Ixx', Ixx_sym, default=Ixx)
        self.register_param('Iyy', Iyy_sym, default=Iyy)
        self.register_param('Izz', Izz_sym, default=Izz)

        I_local = sp.diag(Ixx_sym, Iyy_sym, Izz_sym)
        g_vec = sp.Matrix(g)

        # Differential States
        # 1. Position: r = [x, y, z]
        r = sp.Matrix([sp.Function(f'x_{self.name}')(t),
                      sp.Function(f'y_{self.name}')(t),
                      sp.Function(f'z_{self.name}')(t)])
        # 2. Orientation: R (9 elements)
        R = sp.Matrix([[sp.Function(f'R_{self.name}_{i}_{j}')(t) for j in range(3)] for i in range(3)])
        # 3. Linear Velocity: v = [vx, vy, vz]
        v = sp.Matrix([sp.Function(f'vx_{self.name}')(t),
                      sp.Function(f'vy_{self.name}')(t),
                      sp.Function(f'vz_{self.name}')(t)])
        # 4. Angular Velocity: w = [wx, wy, wz]
        w = sp.Matrix([sp.Function(f'wx_{self.name}')(t),
                      sp.Function(f'wy_{self.name}')(t),
                      sp.Function(f'wz_{self.name}')(t)])

        for r_i in r:
            self.states.append(r_i)
        for i in range(3):
            for j in range(3):
                self.states.append(R[i, j])
        for v_i in v:
            self.states.append(v_i)
        for w_i in w:
            self.states.append(w_i)

        # Port flow variables
        f_p = sp.Matrix([sp.Function(f'f_{self.name}_{i}')(t) for i in range(3)])
        tau_p = sp.Matrix([sp.Function(f'tau_{self.name}_{i}')(t) for i in range(3)])

        # Time derivatives of states
        dr = sp.Matrix([sp.Derivative(r_i, t) for r_i in r])
        dv = sp.Matrix([sp.Derivative(v_i, t) for v_i in v])
        dw = sp.Matrix([sp.Derivative(w_i, t) for w_i in w])
        dR = sp.Matrix([[sp.Derivative(R[i, j], t) for j in range(3)] for i in range(3)])

        # Kinematic Equations
        eq_dr = dr - v
        eq_dR = dR - skew(w) * R

        # Dynamic Equations (Newton-Euler)
        # 1. Newton: m * (dv - g) = f_p
        eq_newton = m_sym * (dv - g_vec) - f_p

        # 2. Euler (in local body frame, mapped to world frame):
        w_local = R.T * w
        dw_local = R.T * dw
        eq_euler = R * (I_local * dw_local + w_local.cross(I_local * w_local)) - tau_p

        # Add equations
        self.equations.extend(list(eq_dr))
        self.equations.extend(list(eq_dR))
        self.equations.extend(list(eq_newton))
        self.equations.extend(list(eq_euler))

        self.ports = {'p': [[f_p, tau_p], [r, R], [v, w]]}


class FixedTranslation(Component):
    def __init__(self, name, r_rel=[0.0, 0.0, 0.0], R_rel=None):
        super().__init__(name)

        if R_rel is None:
            R_rel = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        # Parameters
        r_rel_sym = sp.Matrix([sp.Symbol(f'r_rel_{self.name}_{i}') for i in range(3)])
        for i in range(3):
            self.register_param(f'r_rel_{i}', r_rel_sym[i], default=r_rel[i])

        R_rel_sym = sp.Matrix([[sp.Symbol(f'R_rel_{self.name}_{i}_{j}') for j in range(3)] for i in range(3)])
        for i in range(3):
            for j in range(3):
                self.register_param(f'R_rel_{i}_{j}', R_rel_sym[i, j], default=R_rel[i][j])

        # Port variables
        # Port 1
        f1 = sp.Matrix([sp.Function(f'f1_{self.name}_{i}')(t) for i in range(3)])
        tau1 = sp.Matrix([sp.Function(f'tau1_{self.name}_{i}')(t) for i in range(3)])
        r1 = sp.Matrix([sp.Function(f'r1_{self.name}_{i}')(t) for i in range(3)])
        R1 = sp.Matrix([[sp.Function(f'R1_{self.name}_{i}_{j}')(t) for j in range(3)] for i in range(3)])
        v1 = sp.Matrix([sp.Function(f'v1_{self.name}_{i}')(t) for i in range(3)])
        w1 = sp.Matrix([sp.Function(f'w1_{self.name}_{i}')(t) for i in range(3)])

        # Port 2
        f2 = sp.Matrix([sp.Function(f'f2_{self.name}_{i}')(t) for i in range(3)])
        tau2 = sp.Matrix([sp.Function(f'tau2_{self.name}_{i}')(t) for i in range(3)])
        r2 = sp.Matrix([sp.Function(f'r2_{self.name}_{i}')(t) for i in range(3)])
        R2 = sp.Matrix([[sp.Function(f'R2_{self.name}_{i}_{j}')(t) for j in range(3)] for i in range(3)])
        v2 = sp.Matrix([sp.Function(f'v2_{self.name}_{i}')(t) for i in range(3)])
        w2 = sp.Matrix([sp.Function(f'w2_{self.name}_{i}')(t) for i in range(3)])

        # Connection equations
        # 1. Position relation: r2 = r1 + R1 * r_rel
        eq_pos = r2 - (r1 + R1 * r_rel_sym)

        # 2. Orientation relation: R2 = R1 * R_rel
        eq_rot = R2 - R1 * R_rel_sym

        # 3. Velocity relation: v2 = v1 + w1.cross(R1 * r_rel)
        eq_vel = v2 - (v1 + w1.cross(R1 * r_rel_sym))

        # 4. Angular velocity relation: w2 = w1
        eq_w = w2 - w1

        # 5. Force balance: f1 + f2 = 0
        eq_f = f1 + f2

        # 6. Torque balance: tau1 + tau2 + (R1 * r_rel).cross(f2) = 0
        eq_tau = tau1 + tau2 + (R1 * r_rel_sym).cross(f2)

        self.equations.extend(list(eq_pos))
        self.equations.extend(list(eq_rot))
        self.equations.extend(list(eq_vel))
        self.equations.extend(list(eq_w))
        self.equations.extend(list(eq_f))
        self.equations.extend(list(eq_tau))

        self.ports = {
            'p1': [[f1, tau1], [r1, R1], [v1, w1]],
            'p2': [[f2, tau2], [r2, R2], [v2, w2]]
        }


class RevoluteJoint(Component):
    def __init__(self, name, axis=[0.0, 0.0, 1.0], tau_drive=0.0):
        super().__init__(name)

        # Axis unit vector
        axis_vec = sp.Matrix(axis).normalized()

        # Joint angle and velocity states
        phi = sp.Function(f'phi_{self.name}')(t)
        omega_j = sp.Function(f'omega_j_{self.name}')(t)

        self.states.extend([phi, omega_j])

        # Drive torque parameter
        tau_val_sym = sp.Symbol(f'tau_val_{self.name}')
        self.register_param('tau_drive', tau_val_sym, default=tau_drive)

        # Port variables
        # Port 1
        f1 = sp.Matrix([sp.Function(f'f1_{self.name}_{i}')(t) for i in range(3)])
        tau1 = sp.Matrix([sp.Function(f'tau1_{self.name}_{i}')(t) for i in range(3)])
        r1 = sp.Matrix([sp.Function(f'r1_{self.name}_{i}')(t) for i in range(3)])
        R1 = sp.Matrix([[sp.Function(f'R1_{self.name}_{i}_{j}')(t) for j in range(3)] for i in range(3)])
        v1 = sp.Matrix([sp.Function(f'v1_{self.name}_{i}')(t) for i in range(3)])
        w1 = sp.Matrix([sp.Function(f'w1_{self.name}_{i}')(t) for i in range(3)])

        # Port 2
        f2 = sp.Matrix([sp.Function(f'f2_{self.name}_{i}')(t) for i in range(3)])
        tau2 = sp.Matrix([sp.Function(f'tau2_{self.name}_{i}')(t) for i in range(3)])
        r2 = sp.Matrix([sp.Function(f'r2_{self.name}_{i}')(t) for i in range(3)])
        R2 = sp.Matrix([[sp.Function(f'R2_{self.name}_{i}_{j}')(t) for j in range(3)] for i in range(3)])
        v2 = sp.Matrix([sp.Function(f'v2_{self.name}_{i}')(t) for i in range(3)])
        w2 = sp.Matrix([sp.Function(f'w2_{self.name}_{i}')(t) for i in range(3)])

        # Relative rotation matrix about the specified axis:
        # Standard Rodrigues rotation formula: R = I + sin(phi)*skew(axis) + (1-cos(phi))*skew(axis)^2
        I3 = sp.eye(3)
        K = skew(axis_vec)
        R_rel = I3 + sp.sin(phi) * K + (1 - sp.cos(phi)) * K * K

        # Kinematic of joint angle: d(phi)/dt = omega_j
        eq_j_kin = sp.Derivative(phi, t) - omega_j

        # Equations relating port 1 and port 2
        eq_pos = r2 - r1
        eq_rot = R2 - R1 * R_rel
        eq_vel = v2 - v1
        
        # w2 = w1 + omega_j * R1 * axis_vec
        u_world = R1 * axis_vec
        eq_w = w2 - (w1 + omega_j * u_world)

        # Force and torque balances
        eq_f = f1 + f2
        eq_tau = tau1 + tau2

        # Torque projection on joint axis: u_world.dot(tau2) - tau_drive = 0
        eq_drive = (u_world.T * tau2)[0, 0] - tau_val_sym

        self.equations.append(eq_j_kin)
        self.equations.extend(list(eq_pos))
        self.equations.extend(list(eq_rot))
        self.equations.extend(list(eq_vel))
        self.equations.extend(list(eq_w))
        self.equations.extend(list(eq_f))
        self.equations.extend(list(eq_tau))
        self.equations.append(eq_drive)

        self.ports = {
            'p1': [[f1, tau1], [r1, R1], [v1, w1]],
            'p2': [[f2, tau2], [r2, R2], [v2, w2]]
        }
