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

def get_highest_derivative_order(expr, state_func, t):
    """Find the highest derivative order of state_func in expr with respect to t.
    Returns -1 if not present.
    """
    max_order = -1
    
    def traverse(node):
        nonlocal max_order
        if node == state_func:
            max_order = max(max_order, 0)
            return
        if isinstance(node, sp.Derivative) and node.expr == state_func:
            if all(v == t for v in node.variables):
                max_order = max(max_order, len(node.variables))
            return
        for arg in node.args:
            traverse(arg)
            
    traverse(expr)
    return max_order

def pantelides_pass(dae: SystemDAE) -> SystemDAE:
    """Perform structural index reduction using the Pantelides algorithm.
    Returns a new SystemDAE with reduced index and solved variable matching.
    """
    import networkx as nx
    from networkx.algorithms import bipartite

    t = dae.t
    equations = dae.equations
    
    # Filter out redundant derivative equations (e.g. constraints already present in both position and velocity forms)
    simplified_eqs = [sp.expand(eq) for eq in equations]
    to_remove = set()
    for i in range(len(equations)):
        for j in range(len(equations)):
            if i == j:
                continue
            diff_j = sp.diff(simplified_eqs[j], t)
            if sp.simplify(simplified_eqs[i] - diff_j) == 0 or sp.simplify(simplified_eqs[i] + diff_j) == 0:
                to_remove.add(i)
                break
    equations = [eq for idx, eq in enumerate(equations) if idx not in to_remove]
    
    # Automatically discover all functions of t in the equations to capture all algebraic and differential states
    discovered_states = set(dae.states)
    for eq in equations:
        funcs = eq.find(sp.Function)
        for func in funcs:
            if func != t and func.args == (t,):
                discovered_states.add(func)
    states = sorted(list(discovered_states), key=str)
    
    N = len(equations)
    M = len(states)
    
    # 1. Build original_h matrix: highest derivative of states[j] in equations[k]
    original_h = []
    for eq in equations:
        row = []
        for state in states:
            order = get_highest_derivative_order(eq, state, t)
            row.append(order)
        original_h.append(row)
        
    # 2. Run Pantelides algorithm to find differentiation index d[k] for each equation k
    d = [0] * N
    for i in range(N):
        while True:
            # Compute c[j] = max_{k <= i} (original_h[k][j] + d[k])
            c = []
            for j in range(M):
                max_order = -1
                for k in range(i + 1):
                    orig_order = original_h[k][j]
                    if orig_order >= 0:
                        max_order = max(max_order, orig_order + d[k])
                c.append(max_order)
                
            # Build edges for equations 0..i
            edges = {}
            for k in range(i + 1):
                edges[k] = []
                for j in range(M):
                    orig_order = original_h[k][j]
                    if orig_order >= 0 and (orig_order + d[k] == c[j]):
                        edges[k].append(j)
                        
            # Build NetworkX bipartite graph for equations 0..i
            B = nx.Graph()
            B.add_nodes_from([f"eq_{k}" for k in range(i + 1)], bipartite=0)
            B.add_nodes_from([f"var_{j}" for j in range(M)], bipartite=1)
            for k in range(i + 1):
                for j in edges[k]:
                    B.add_edge(f"eq_{k}", f"var_{j}")
                    
            # Try to match equations 0..i
            matching = bipartite.maximum_matching(B, top_nodes=[f"eq_{k}" for k in range(i + 1)])
            
            # Check if all equations 0..i are matched
            unmatched_eqs = [f"eq_{k}" for k in range(i + 1) if f"eq_{k}" not in matching]
            
            if not unmatched_eqs:
                break  # Successfully matched all equations 0..i, proceed to next equation i+1
                
            # If not all equations matched, we construct the directed alternating graph to find deficient sets
            D = nx.DiGraph()
            for u, v in B.edges():
                if u.startswith("eq_"):
                    eq_node, var_node = u, v
                else:
                    eq_node, var_node = v, u
                    
                if matching.get(eq_node) == var_node:
                    D.add_edge(var_node, eq_node)
                else:
                    D.add_edge(eq_node, var_node)
                    
            # Find all reachable nodes in D starting from unmatched equations
            visited = set(unmatched_eqs)
            queue = list(unmatched_eqs)
            while queue:
                curr = queue.pop(0)
                if curr in D:
                    for neighbor in D.neighbors(curr):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                            
            # Increment differentiation index for all visited equations
            visited_eq_indices = [int(node.split("_")[1]) for node in visited if node.startswith("eq_")]
            for v_eq in visited_eq_indices:
                d[v_eq] += 1
                
    # 3. Construct the new DAE and differentiate equations
    new_dae = dae.copy_structure()
    
    # Differentiate equations as required by d[k]
    active_equations = []
    all_equations = []
    
    for k, eq in enumerate(equations):
        curr_eq = eq
        all_equations.append(curr_eq)
        for step in range(d[k]):
            curr_eq = sp.diff(curr_eq, t)
            all_equations.append(curr_eq)
        active_equations.append(curr_eq)
        
    new_dae.equations = all_equations
    new_dae.active_equations = active_equations
    new_dae.differentiation_indices = d
    
    # 4. Identify final c_j values across all equations
    final_c = []
    for j in range(M):
        max_order = -1
        for k in range(N):
            orig_order = original_h[k][j]
            if orig_order >= 0:
                max_order = max(max_order, orig_order + d[k])
        final_c.append(max_order)
        
    # 5. Extract solved variables and differential states
    solved_variables = []
    differential_states = []
    
    for j, state in enumerate(states):
        c_j = final_c[j]
        if c_j < 0:
            continue
        if c_j == 0:
            solved_var = state
        else:
            solved_var = sp.Derivative(state, t, c_j)
        solved_variables.append(solved_var)
        
        for l in range(c_j):
            if l == 0:
                diff_state = state
            else:
                diff_state = sp.Derivative(state, t, l)
            if diff_state not in differential_states:
                differential_states.append(diff_state)
                
    new_dae.solved_variables = solved_variables
    new_dae.states = differential_states
    
    # 6. Rebuild final matching map using NetworkX
    edges = {}
    for k in range(N):
        edges[k] = []
        for j in range(M):
            orig_order = original_h[k][j]
            if orig_order >= 0 and (orig_order + d[k] == final_c[j]):
                edges[k].append(j)
                
    B = nx.Graph()
    B.add_nodes_from([f"eq_{k}" for k in range(N)], bipartite=0)
    B.add_nodes_from([f"var_{j}" for j in range(M)], bipartite=1)
    for k in range(N):
        for j in edges[k]:
            B.add_edge(f"eq_{k}", f"var_{j}")
            
    matching = bipartite.maximum_matching(B, top_nodes=[f"eq_{k}" for k in range(N)])
    
    new_dae.matching = {}
    for k in range(N):
        matched_var = matching.get(f"eq_{k}")
        if matched_var:
            var_idx = int(matched_var.split("_")[1])
            state = states[var_idx]
            c_j = final_c[var_idx]
            if c_j == 0:
                solved_var = state
            else:
                solved_var = sp.Derivative(state, t, c_j)
            new_dae.matching[k] = solved_var
        
    # Populate derivatives map for the differential states
    new_dae.derivatives = {}
    for j, state in enumerate(states):
        c_j = final_c[j]
        for l in range(c_j):
            if l == 0:
                curr = state
            else:
                curr = sp.Derivative(state, t, l)
            
            if l + 1 == 1:
                nxt = sp.Derivative(state, t)
            else:
                nxt = sp.Derivative(state, t, l + 1)
            new_dae.derivatives[curr] = nxt
            
    return new_dae

def tearing_pass(dae: SystemDAE) -> SystemDAE:
    """Symbolically solves the active equations for all solved variables (tearing pass).
    If successful, stores the mapping in solved_assignments on the DAE.
    """
    new_dae = dae.copy_structure()
    try:
        solutions = sp.solve(dae.active_equations, dae.solved_variables, dict=True)
        if solutions and len(solutions[0]) == len(dae.solved_variables):
            new_dae.solved_assignments = solutions[0]
        else:
            new_dae.solved_assignments = {}
    except Exception as e:
        new_dae.solved_assignments = {}
        
    return new_dae

def simplification_pass(dae: SystemDAE) -> SystemDAE:
    """Simplifies the solved assignments symbolically using SymPy's simplify.
    Also performs elimination of redundant states/variables and separates state derivative
    assignments into a dedicated ode_assignments map.
    """
    new_dae = dae.copy_structure()
    new_dae.solved_assignments = {}
    new_dae.ode_assignments = {}
    
    if not dae.solved_assignments:
        return new_dae
        
    t = dae.t
    
    # 1. Find state derivative definitions of the form Derivative(A, t) = B
    # where A and B are both functions of t.
    defs = {}
    for eq in dae.equations:
        eq_expanded = sp.expand(eq)
        # Find all first derivatives of functions in the equation
        derivs = [d for d in eq_expanded.find(sp.Derivative) if len(d.variables) == 1 and d.variables[0] == t]
        if len(derivs) == 1:
            deriv = derivs[0]
            try:
                sol = sp.solve(eq_expanded, deriv)
                if sol and len(sol) == 1:
                    rhs = sol[0]
                    # Check if rhs is a function of t (excluding t itself)
                    if isinstance(rhs, sp.Function) and rhs != t and rhs.args == (t,):
                        defs[deriv] = rhs
            except Exception:
                pass

    # 2. Substitution mapping:
    # If Derivative(A, t) = B, then replace Derivative(A, t) with B.
    # Also, Derivative(A, (t, 2)) becomes Derivative(B, t).
    sub_dict = {}
    for deriv, state_var in defs.items():
        sub_dict[deriv] = state_var
        # Also handle second derivative:
        second_deriv = sp.Derivative(deriv.expr, t, 2)
        sub_dict[second_deriv] = sp.Derivative(state_var, t)

    # 3. Apply substitutions to values and simplify solved_assignments
    simplified = {}
    for var, expr in dae.solved_assignments.items():
        # Apply substitutions to the value (expr) but keep key (var) as is
        new_expr = expr.subs(sub_dict)
        # Simplify the expression
        simplified[var] = sp.simplify(new_expr)

    new_dae.solved_assignments = simplified

    # 4. Update states: eliminate redundant states (those of the form Derivative(A, t) that are mapped to another state)
    redundant_states = set()
    for deriv in defs.keys():
        if deriv in dae.states:
            redundant_states.add(deriv)

    new_states = [s for s in dae.states if s not in redundant_states]
    new_dae.states = new_states

    # 5. Extract state derivative assignments for the ODE integration
    ode_assignments = {}
    for state in new_states:
        # The derivative we want to assign:
        state_deriv = sp.Derivative(state, t)
        
        # If the state_deriv is defined in defs, use that definition
        if state_deriv in defs:
            ode_assignments[state_deriv] = defs[state_deriv]
        else:
            # Otherwise, look for it in the simplified solved assignments
            found = False
            for var, expr in simplified.items():
                if var == state_deriv:
                    ode_assignments[state_deriv] = expr
                    found = True
                    break
            
            # If not found in solved assignments, check if the derivative itself is a state
            if not found:
                if state_deriv in new_states:
                    ode_assignments[state_deriv] = state_deriv

    new_dae.ode_assignments = ode_assignments
    return new_dae