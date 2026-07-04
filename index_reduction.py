"""
index_reduction.py
==================
CasADi-based DAE index reduction and tearing for Braid.

Pipeline
--------
1. pantelides_pass(dae)  — structural index reduction via Pantelides algorithm.
   Input:  CasadiDAE with equations, x_vars, xdot_vars, p_vars
   Output: CasadiDAE with active_equations, solved_variables,
           differentiation_indices, derivative_chain extended

2. tearing_pass(dae)     — linear algebraic solve to get explicit ODE RHS.
   Input:  CasadiDAE from pantelides_pass
   Output: CasadiDAE with ode_rhs (explicit xdot = f(x,p,t) per state)

Key differences from the SymPy version
---------------------------------------
- Incidence matrix built from ca.jacobian() sparsity (no get_highest_derivative_order)
- Differentiation uses CasADi chain rule: d(F)/dt = Σ_i ∂F/∂x_i * xdot_i
  extended to higher-order derivatives via derivative_chain
- Tearing uses ca.solve(J, -b) for linear systems; raises an error if nonlinear
  (in practice all standard physical components produce linear systems after
  index reduction — contact the maintainers if you hit this)
"""

import numpy as np
import casadi as ca
import networkx as nx
from networkx.algorithms import bipartite

from casadi_dae import CasadiDAE


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sym_name(sx: ca.SX) -> str:
    """Return the string name of a scalar CasADi SX symbol."""
    return sx.name()


def _build_incidence(equations: list, variables: list) -> np.ndarray:
    """
    Build a binary incidence matrix H of shape (n_eq × n_var).
    H[i, j] = 1  iff  variables[j] appears in equations[i].

    Uses CasADi's jacobian sparsity — faster than symbolic expansion.
    """
    n_eq  = len(equations)
    n_var = len(variables)
    H = np.zeros((n_eq, n_var), dtype=int)

    if n_var == 0 or n_eq == 0:
        return H

    # Stack all variables into a single vector for batch jacobian
    var_vec = ca.vertcat(*variables)

    for i, eq in enumerate(equations):
        try:
            J = ca.jacobian(ca.SX(eq), var_vec)
            sp = J.sparsity()
            # sp.get_col() returns column indices of nonzero entries
            for col in sp.get_col():
                H[i, col] = 1
        except Exception:
            # If jacobian fails, assume equation involves all variables
            H[i, :] = 1

    return H


def _differentiate_eq(eq: ca.SX, all_vars: list, all_derivs: list) -> ca.SX:
    """
    Compute d(eq)/dt via the chain rule:
        d(eq)/dt = Σ_i  ∂eq/∂v_i  *  dv_i/dt

    where all_vars  = [x0, x1, ..., xdot0, xdot1, ...]
    and   all_derivs = [xdot0, xdot1, ..., xddot0, xddot1, ...]

    all_vars and all_derivs must be the same length and paired:
        all_derivs[i] is the time derivative of all_vars[i].
    """
    result = ca.SX(0)
    var_vec  = ca.vertcat(*all_vars)
    try:
        J = ca.jacobian(ca.SX(eq), var_vec)   # 1 × n_var
        for j, deriv_sym in enumerate(all_derivs):
            j_entry = J[0, j]
            if not ca.is_equal(j_entry, ca.SX(0)):
                result = result + j_entry * deriv_sym
    except Exception:
        raise RuntimeError(
            f"Failed to differentiate equation '{eq}' w.r.t. variables. "
            "Ensure all symbolic variables are registered via add_state() / register_param()."
        )
    return result


def _extend_derivative_chain(
    dae: CasadiDAE,
    order: int,
) -> tuple[list, list]:
    """
    Extend the derivative chain to support up to `order` levels of differentiation.

    Returns (all_vars, all_derivs) where:
        all_vars[i]   is a variable at some differentiation level
        all_derivs[i] is its time derivative

    For order=1:  all_vars = [x, xdot],  all_derivs = [xdot, xddot]
    etc.
    """
    # Build chain: x → xdot → xddot → ...
    # dae.derivative_chain already has x→xdot from to_dae()
    chain = dict(dae.derivative_chain)   # {sym: d_sym}

    # Extend to required order
    current_level = list(dae.xdot_vars)
    for lvl in range(order):
        next_level = []
        for sym in current_level:
            if sym not in chain:
                # Create a new higher-order derivative symbol
                new_sym = ca.SX.sym(f'{sym.name()}_dot')
                chain[sym] = new_sym
            next_level.append(chain[sym])
        current_level = next_level

    # Update dae derivative chain
    dae.derivative_chain.update(chain)

    # Build the paired lists for _differentiate_eq
    all_vars   = []
    all_derivs = []
    visited    = set()

    def _add_pair(v, d):
        key = str(v)
        if key not in visited:
            visited.add(key)
            all_vars.append(v)
            all_derivs.append(d)

    for x, xdot in zip(dae.x_vars, dae.xdot_vars):
        _add_pair(x, xdot)
    for sym, deriv in chain.items():
        if str(sym) not in visited:
            _add_pair(sym, deriv)

    return all_vars, all_derivs


# ─────────────────────────────────────────────────────────────────────────────
# Pantelides Pass
# ─────────────────────────────────────────────────────────────────────────────

def pantelides_pass(dae: CasadiDAE) -> CasadiDAE:
    """
    Perform structural index reduction using the Pantelides algorithm.

    The algorithm:
    1. Builds an incidence matrix over the original equations and the
       union of (state, derivative) variables.
    2. Iteratively differentiates equations that prevent a perfect matching
       in the bipartite graph, until a perfect matching exists.
    3. Returns a new CasadiDAE with:
       - active_equations: the final set of (possibly differentiated) equations
       - solved_variables: matched variable per active equation
       - differentiation_indices: how many times each original eq was differentiated
    """
    new_dae = dae.copy_structure()

    equations = list(dae.equations)

    # ── Step 0: Convert purely algebraic states to algebraic variables ──────────
    # If a state derivative in dae.xdot_vars does not appear in any equation,
    # then it has no differential dynamics and is actually a purely algebraic variable.
    import casadi as ca
    active_xdot_names = set()
    for eq in equations:
        for v in ca.symvar(ca.SX(eq)):
            active_xdot_names.add(v.name())

    real_x_vars = []
    real_xdot_vars = []
    real_state_names = []
    for x, xdot in zip(dae.x_vars, dae.xdot_vars):
        if str(xdot) in active_xdot_names:
            real_x_vars.append(x)
            real_xdot_vars.append(xdot)
            real_state_names.append(str(x))

    new_dae.x_vars = real_x_vars
    new_dae.xdot_vars = real_xdot_vars
    new_dae.state_names = real_state_names

    # ── Step 0: deduplicate structurally redundant equations ─────────────────
    # Two equations are redundant if one is a scalar multiple of the other
    # (detected via incidence pattern equality). Quick structural check only.
    all_vars_initial = list(dae.x_vars) + list(dae.xdot_vars) + list(dae.p_vars)
    H_init = _build_incidence(equations, all_vars_initial)
    to_remove = set()
    for i in range(len(equations)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(equations)):
            if j in to_remove:
                continue
            if np.array_equal(H_init[i], H_init[j]):
                # Same incidence pattern — likely structurally redundant
                # Verify by checking if i ≈ j or i ≈ -j via CasADi simplification
                try:
                    diff = ca.simplify(ca.SX(equations[i]) - ca.SX(equations[j]))
                    if ca.is_equal(diff, ca.SX(0)):
                        to_remove.add(j)
                        continue
                    diff2 = ca.simplify(ca.SX(equations[i]) + ca.SX(equations[j]))
                    if ca.is_equal(diff2, ca.SX(0)):
                        to_remove.add(j)
                except Exception:
                    pass
    equations = [eq for k, eq in enumerate(equations) if k not in to_remove]

    N = len(equations)

    # ── Step 1: collect all symbolic variables (states + derivatives + params) ─
    # For the bipartite graph we only match over state derivative and algebraic variables,
    # not the integrated states (which are known inputs) and not parameters.
    import casadi as ca
    all_syms = set()
    for eq in equations:
        all_syms.update(ca.symvar(eq))
    param_names_set = set(dae.param_names)
    state_vars = [v for v in all_syms if v.name() not in param_names_set]
    # Ensure they are sorted or at least deterministic
    state_vars = sorted(state_vars, key=lambda s: s.name())
    M = len(state_vars)

    # ── Step 2: build initial incidence (states + xdots only) ───────────────
    H = _build_incidence(equations, state_vars)

    # differentiation count per original equation
    d = [0] * N

    # Working copies of equations (will be differentiated)
    working_eqs = list(equations)

    # ── Step 3: Pantelides main loop ─────────────────────────────────────────
    max_diff_order = 5   # safety limit

    for i in range(N):
        for _attempt in range(max_diff_order):
            # Build bipartite graph for equations 0..i
            # To ensure state derivatives (xdot_vars) are solved directly, we can prioritize matching them.
            # networkx maximum_matching doesn't support weights directly, but we can search for a matching
            # that prefers state derivative matching by doing a two-stage matching or sorting.
            # Even simpler: we can match over xdot_vars first, then match remaining equations over other variables.
            B = nx.Graph()
            eq_nodes  = [f'eq_{k}' for k in range(i + 1)]
            var_nodes = [f'var_{j}' for j in range(M)]
            B.add_nodes_from(eq_nodes,  bipartite=0)
            B.add_nodes_from(var_nodes, bipartite=1)

            xdot_names_set = set(str(s) for s in dae.xdot_vars)

            for k in range(i + 1):
                for j in range(M):
                    if H[k, j] == 1:
                        # Prioritize state derivatives by matching them if possible
                        B.add_edge(f'eq_{k}', f'var_{j}')

            # Use networkx maximum_matching
            matching = bipartite.maximum_matching(B, top_nodes=eq_nodes)
            unmatched = [f'eq_{k}' for k in range(i + 1) if f'eq_{k}' not in matching]

            if not unmatched:
                break  # all equations 0..i matched — proceed

            # Find reachable set via alternating path from unmatched equations
            D = nx.DiGraph()
            for u, v in B.edges():
                if u.startswith('eq_'):
                    eq_n, var_n = u, v
                else:
                    eq_n, var_n = v, u
                if matching.get(eq_n) == var_n:
                    D.add_edge(var_n, eq_n)
                else:
                    D.add_edge(eq_n, var_n)

            visited = set(unmatched)
            queue = list(unmatched)
            while queue:
                curr = queue.pop(0)
                if curr in D:
                    for nbr in D.successors(curr):
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)

            # Differentiate all reachable equations once
            visited_eq_indices = [
                int(node.split('_')[1]) for node in visited if node.startswith('eq_')
            ]

            # Extend derivative chain to handle one more order
            all_vars_chain, all_derivs_chain = _extend_derivative_chain(dae, max(d) + 2)

            # Rebuild state_vars to include any new higher-order derivative symbols
            state_var_names = set(str(s) for s in state_vars)
            new_syms = [s for s in all_derivs_chain if str(s) not in state_var_names]
            state_vars = state_vars + new_syms
            M = len(state_vars)

            # Resize H
            H_new = np.zeros((N, M), dtype=int)
            H_new[:, :H.shape[1]] = H
            H = H_new

            for k in visited_eq_indices:
                d[k] += 1
                new_eq = _differentiate_eq(
                    working_eqs[k], all_vars_chain, all_derivs_chain
                )
                working_eqs[k] = new_eq
                # Update incidence row k
                row = np.zeros(M, dtype=int)
                if state_vars:
                    sv = ca.vertcat(*state_vars)
                    try:
                        J = ca.jacobian(ca.SX(new_eq), sv)
                        sp = J.sparsity()
                        for col in sp.get_col():
                            row[col] = 1
                    except Exception:
                        row[:] = 1
                H[k, :] = row

        else:
            raise RuntimeError(
                f"Pantelides: could not find a perfect matching for equation {i} "
                f"after {max_diff_order} differentiations. "
                "The system may be over- or under-determined."
            )

    # ── Step 4: build final active equations and matching ────────────────────
    # The active equations are the final (possibly differentiated) working_eqs.
    # We need to find what each active equation solves for. Prioritize matching
    # state derivatives (xdot_vars) by assigning them higher edge weights.
    # Exclude integrated states (state_names) from matching, as they are inputs.
    B_final = nx.Graph()
    eq_nodes  = [f'eq_{k}' for k in range(N)]
    var_nodes = [f'var_{j}' for j in range(M)]
    B_final.add_nodes_from(eq_nodes, bipartite=0)
    B_final.add_nodes_from(var_nodes, bipartite=1)

    xdot_names_set = set(str(s) for s in new_dae.xdot_vars)
    state_names_set = set(new_dae.state_names)

    for k in range(N):
        for j in range(M):
            if H[k, j] == 1:
                var_name = state_vars[j].name()
                if var_name in state_names_set:
                    continue  # Do not match integrated states
                # Assign weight 100 to state derivatives, 1 to others
                weight = 100.0 if var_name in xdot_names_set else 1.0
                B_final.add_edge(f'eq_{k}', f'var_{j}', weight=weight)

    # Use networkx max_weight_matching to compute matching
    weight_matching = nx.max_weight_matching(B_final)

    # Convert the undirected set of matching edges back to final_matching dict
    final_matching = {}
    for u, v in weight_matching:
        final_matching[u] = v
        final_matching[v] = u

    # Build solved_variables list in equation order
    solved_variables = []
    for k in range(N):
        matched_var = final_matching.get(f'eq_{k}')
        if matched_var:
            var_idx = int(matched_var.split('_')[1])
            solved_variables.append(state_vars[var_idx])
        else:
            solved_variables.append(None)

    new_dae.active_equations      = working_eqs
    new_dae.solved_variables      = solved_variables
    new_dae.differentiation_indices = d
    new_dae.derivative_chain      = dict(dae.derivative_chain)

    # Only extend x_vars if a state derivative was differentiated (Pantelides chain extension)
    # But keep the original x_vars/xdot_vars clean from algebraic variables.
    # Algebraic variables are solved and substituted away by tearing, not integrated.
    pass

    return new_dae


# ─────────────────────────────────────────────────────────────────────────────
# Tearing Pass
# ─────────────────────────────────────────────────────────────────────────────

def tearing_pass(dae: CasadiDAE) -> CasadiDAE:
    """
    Solve the active equations for the matched solved variables using
    CasADi's linear solve: ca.solve(J, -b).

    For each active equation F_k(solved_vars, known) = 0:
        J = ∂F_k/∂(solved_var_k)   (scalar or vector, depending on coupling)
        b = F_k evaluated at solved_var_k = 0

    If J is independent of solved_vars (system is linear), this gives the
    exact solution. Otherwise, raises a NotImplementedError.

    Returns a new CasadiDAE with:
        ode_rhs[i] = explicit expression for xdot_vars[i]
        alg_assignments = {algebraic_var: expression, ...}
    """
    new_dae = dae.copy_structure()

    active_eqs   = dae.active_equations
    solved_vars  = dae.solved_variables

    # Pair up: (equation, variable it solves for)
    pairs = [(eq, sv) for eq, sv in zip(active_eqs, solved_vars) if sv is not None]

    # Solve sequentially: substitute already-solved variables before each solve
    assignments = {}   # {ca.SX symbol: ca.SX expression}

    def _substitute_assignments(expr: ca.SX) -> ca.SX:
        """Apply all current assignments as substitutions."""
        if not assignments:
            return expr
        # Substitute using actual CasADi symbols as keys
        result = expr
        for sym_key, val in assignments.items():
            result = ca.substitute(result, sym_key, val)
        return result

    for eq_raw, var in pairs:
        # Apply prior assignments to reduce the equation
        eq = _substitute_assignments(ca.SX(eq_raw))

        # Compute J = ∂eq/∂var  and  b = eq|_{var=0}
        J = ca.jacobian(eq, var)          # should be constant if linear
        b = ca.substitute(eq, var, ca.SX(0))

        # Check linearity: J should not depend on var
        J_check = ca.jacobian(J, var)
        if not ca.is_equal(ca.simplify(J_check), ca.SX(0)):
            raise NotImplementedError(
                f"Tearing: equation for '{var.name()}' is nonlinear in that variable. "
                "Nonlinear algebraic loops require the 'casadi' backend (IDAS solver) "
                "and cannot be lowered to explicit form."
            )

        # Solve: J * var = -b  →  var = -b / J  (scalar case)
        try:
            sol = ca.solve(J, -b)
        except Exception as e:
            raise RuntimeError(
                f"Tearing: ca.solve failed for variable '{var.name()}': {e}"
            )

        assignments[var] = sol

    # ── Map assignments back to ode_rhs ──────────────────────────────────────
    # We want ode_rhs[i] = expression for the original xdot_vars[i]
    ode_rhs = []
    alg_assignments = {}

    for var, expr in assignments.items():
        alg_assignments[var.name()] = _fully_substitute(expr, assignments)

    # First round of ODE RHS extraction
    temp_ode_rhs = {}
    for i, (x, xdot) in enumerate(zip(dae.x_vars, dae.xdot_vars)):
        xdot_name = str(xdot)
        if xdot_name in alg_assignments:
            rhs = alg_assignments[xdot_name]
            rhs = _fully_substitute(rhs, assignments)
            temp_ode_rhs[xdot] = rhs
        else:
            found = False
            for var_name, expr in alg_assignments.items():
                if var_name == xdot_name:
                    temp_ode_rhs[xdot] = expr
                    found = True
                    break
            if not found:
                for eq_raw in list(active_eqs) + list(dae.equations):
                    eq_expr = ca.SX(eq_raw)
                    if xdot_name in [v.name() for v in ca.symvar(eq_expr)]:
                        J_xdot = ca.jacobian(eq_expr, xdot)
                        J_xdot_check = ca.jacobian(J_xdot, xdot)
                        if ca.is_equal(ca.simplify(J_xdot_check), ca.SX(0)) and not ca.is_equal(ca.simplify(J_xdot), ca.SX(0)):
                            rest = ca.substitute(eq_expr, xdot, ca.SX(0))
                            sol_candidate = -rest / J_xdot
                            sol_candidate = _fully_substitute(sol_candidate, assignments)
                            temp_ode_rhs[xdot] = sol_candidate
                            assignments[xdot] = sol_candidate
                            alg_assignments[xdot_name] = sol_candidate
                            found = True
                            break
            if not found:
                raise RuntimeError(
                    f"Tearing: no assignment found for state derivative '{xdot_name}'."
                )

    # Resolve remaining state derivatives in RHS expressions by substituting them
    # with their solved ODE expressions (breaks algebraic loops)
    for i, (x, xdot) in enumerate(zip(dae.x_vars, dae.xdot_vars)):
        rhs = temp_ode_rhs[xdot]
        # Substitute other state derivatives with their solved expressions
        for other_xdot, other_rhs in temp_ode_rhs.items():
            if not ca.is_equal(xdot, other_xdot):
                rhs = ca.substitute(rhs, other_xdot, other_rhs)
        # Substitute all other algebraic assignments
        rhs = _fully_substitute(rhs, assignments)
        ode_rhs.append(rhs)

    new_dae.ode_rhs         = ode_rhs
    new_dae.alg_assignments = alg_assignments
    return new_dae


def _fully_substitute(expr: ca.SX, assignments: dict, max_rounds: int = 20) -> ca.SX:
    """
    Repeatedly substitute assignments into expr until convergence.
    This resolves chains of algebraic dependencies.
    """
    result = expr
    for _ in range(max_rounds):
        prev_str = str(result)
        for sym_key, val in assignments.items():
            result = ca.substitute(result, sym_key, val)
        if str(result) == prev_str:
            break
    return result