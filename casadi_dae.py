"""
casadi_dae.py
=============
Core data structure for Braid's CasADi-based DAE representation.

Replaces sym_dae.py (SymPy-based SystemDAE).

States are CasADi SX scalar symbols. Each state x_i has a paired derivative
symbol xdot_i. Equations are written as residual expressions (= 0).

After index reduction and tearing, `ode_rhs` holds the explicit right-hand side:
    xdot_i = ode_rhs[i](x, p, t)
as a list of ca.SX scalar expressions.
"""

import casadi as ca


class CasadiDAE:
    """
    Symbolic DAE representation using CasADi SX.

    Lifecycle:
        1. Built by System.to_dae() — populates x_vars, xdot_vars, p_vars,
           equations, state_names, param_names.
        2. pantelides_pass() — adds active_equations, solved_variables,
           differentiation_indices, derivative_chain.
        3. tearing_pass() — adds ode_rhs (explicit xdot assignments).
    """

    def __init__(self):
        # ── Symbolic variable lists (scalars) ──────────────────────────────
        self.x_vars: list       = []   # ca.SX scalars — state variables
        self.xdot_vars: list    = []   # ca.SX scalars — derivative symbols (paired)
        self.p_vars: list       = []   # ca.SX scalars — parameter symbols
        self.t: ca.SX           = ca.SX.sym('t')   # time

        # ── Equations (residuals = 0) ───────────────────────────────────────
        # Contains both component equations and Node() connection equations.
        # Elements may involve any of x_vars, xdot_vars, or p_vars.
        self.equations: list    = []   # list of ca.SX scalar expressions

        # ── Name / metadata ────────────────────────────────────────────────
        self.state_names: list  = []   # str names matching x_vars order
        self.xdot_names: list   = []   # str names matching xdot_vars order
        self.param_names: list  = []   # str names matching p_vars order
        self.param_meta: dict   = {}   # {param_name: {component, default, ...}}
        self.components: list   = []   # [{name, domain}, ...]
        self.sensor_mappings: dict = {}

        # ── After index reduction (pantelides_pass) ─────────────────────────
        self.active_equations: list     = []  # reduced set of ca.SX residuals
        self.solved_variables: list     = []  # ca.SX vars solved by tearing
        self.differentiation_indices: list = []  # int per original equation
        # derivative_chain: maps x_sym → xdot_sym → xddot_sym ...
        # Used so the Pantelides differentiator can produce higher-order terms.
        self.derivative_chain: dict     = {}  # {ca.SX: ca.SX}  x→xdot, xdot→xddot
        self.invariants: list           = []  # list of ca.SX original equations differentiated

        # ── After tearing (tearing_pass) ────────────────────────────────────
        # ode_rhs[i] is the explicit expression for d(x_vars[i])/dt
        # as a ca.SX scalar in terms of x_vars, p_vars, and t.
        self.ode_rhs: list              = []   # list of ca.SX (one per state)

        # Algebraic assignments from tearing (non-state variables)
        # {ca.SX solved_var: ca.SX expression}
        self.alg_assignments: dict      = {}

        # DAE implicit residuals (residuals = 0). Used when model_type == 'DAE'
        self.residuals: list            = []


    # ── Stacked vector accessors ────────────────────────────────────────────

    @property
    def x(self) -> ca.SX:
        """Stacked state vector (n_states × 1)."""
        if not self.x_vars:
            return ca.SX(0, 1)
        return ca.vertcat(*self.x_vars)

    @property
    def xdot(self) -> ca.SX:
        """Stacked derivative vector (n_states × 1)."""
        if not self.xdot_vars:
            return ca.SX(0, 1)
        return ca.vertcat(*self.xdot_vars)

    @property
    def p(self) -> ca.SX:
        """Stacked parameter vector (n_params × 1)."""
        if not self.p_vars:
            return ca.SX(0, 1)
        return ca.vertcat(*self.p_vars)

    @property
    def ode_rhs_vec(self) -> ca.SX:
        """Stacked ODE RHS vector (n_states × 1). Only valid after tearing."""
        if not self.ode_rhs:
            return ca.SX(0, 1)
        return ca.vertcat(*self.ode_rhs)

    # ── Utility ─────────────────────────────────────────────────────────────

    def n_states(self) -> int:
        return len(self.x_vars)

    def n_params(self) -> int:
        return len(self.p_vars)

    def state_index(self, name: str) -> int:
        return self.state_names.index(name)

    def param_index(self, name: str) -> int:
        return self.param_names.index(name)

    def get_param_defaults(self) -> list:
        """Returns parameter default values in p_vars order."""
        return [self.param_meta[n]['default'] for n in self.param_names]

    def copy_structure(self) -> 'CasadiDAE':
        """Shallow-copy of structural fields (for pipeline passes)."""
        new = CasadiDAE()
        new.x_vars               = list(self.x_vars)
        new.xdot_vars            = list(self.xdot_vars)
        new.p_vars               = list(self.p_vars)
        new.t                    = self.t
        new.equations            = list(self.equations)
        new.state_names          = list(self.state_names)
        new.xdot_names           = list(self.xdot_names)
        new.param_names          = list(self.param_names)
        new.param_meta           = dict(self.param_meta)
        new.components           = [dict(c) for c in self.components]
        new.sensor_mappings      = dict(self.sensor_mappings)
        new.active_equations     = list(self.active_equations)
        new.solved_variables     = list(self.solved_variables)
        new.differentiation_indices = list(self.differentiation_indices)
        new.derivative_chain     = dict(self.derivative_chain)
        new.invariants           = list(self.invariants)
        new.ode_rhs              = list(self.ode_rhs)
        new.alg_assignments      = dict(self.alg_assignments)
        new.residuals            = list(self.residuals)
        return new


    def __repr__(self) -> str:
        return (
            f"CasadiDAE(states={self.state_names}, "
            f"params={self.param_names}, "
            f"n_equations={len(self.equations)}, "
            f"n_active={len(self.active_equations)}, "
            f"ode_ready={len(self.ode_rhs) == len(self.x_vars)}, "
            f"dae_ready={len(self.residuals) > 0})"
        )

