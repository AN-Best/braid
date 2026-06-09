import sympy as sp

class SystemDAE:
    def __init__(self):
        self.t = sp.Symbol('t')
        self.states = []       # List of sympy Functions of t
        self.derivatives = {}  # Map of state to its derivative
        self.equations = []    # List of sympy expressions (implicitly = 0)
        self.params = []       # List of sympy Symbols
        self.active_equations = []          # List of active sympy expressions
        self.solved_variables = []          # List of solved sympy expressions/variables
        self.matching = {}                  # Map of active equation index to solved variable
        self.differentiation_indices = []  # List of differentiation counts per original equation
        self.components = []                # List of dictionaries containing component metadata
        self.param_meta = {}                # Dict mapping parameter symbol srepr to metadata dict
        self.solved_assignments = {}        # Dict mapping solved variable to explicit expression (from tearing)
        self.ode_assignments = {}           # Dict mapping state derivative to simplified explicit expression

    def copy_structure(self):
        new_dae = SystemDAE()
        new_dae.t = self.t
        new_dae.states = list(self.states)
        new_dae.derivatives = dict(self.derivatives)
        new_dae.equations = list(self.equations)
        new_dae.params = list(self.params)
        new_dae.active_equations = list(self.active_equations)
        new_dae.solved_variables = list(self.solved_variables)
        new_dae.matching = dict(self.matching)
        new_dae.differentiation_indices = list(self.differentiation_indices)
        new_dae.components = [dict(c) for c in self.components]
        new_dae.param_meta = dict(self.param_meta)
        new_dae.solved_assignments = dict(self.solved_assignments)
        new_dae.ode_assignments = dict(self.ode_assignments)
        return new_dae
