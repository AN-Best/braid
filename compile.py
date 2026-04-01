import casadi as ca

def Compile(system):

    all_state = system.states
    all_params = system.params
    all_ode_eqs = system.ode_eqs
    all_alg_eqs = system.alg_eqs + system.connection_alg_eqs

    ode_rhs = []
    for s in all_state:
        ode_rhs.append(all_ode_eqs[s])

    all_syms = set()
    for eq in all_alg_eqs:
        all_syms |= set(ca.symvar(eq))

    x_impl = ca.vertcat(*all_state)
    dx_syms = [ca.SX.sym(f'd{s.name()}') for s in all_state]
    dx_impl = ca.vertcat(*dx_syms)

    kinematic_residuals = [dx - rhs for dx, rhs in zip(dx_syms, ode_rhs)]

    alg_vars = all_syms - set(all_state) - set(all_params)
    alg_vars = list(alg_vars)

    dae = {
        'x_impl': x_impl,
        'dx_impl': dx_impl,
        'z': ca.vertcat(*alg_vars),
        'p': ca.vertcat(*all_params),
        'alg': ca.vertcat(*all_alg_eqs + kinematic_residuals),
    }

    dae_reduced = ca.dae_reduce_index(dae)

    dae_reduced, stats = ca.dae_reduce_index(dae)
    semi_expl, phi_fn, _ = ca.dae_map_semi_expl(dae_reduced, dae_reduced)

    # extract semi-explicit components
    z_sym = semi_expl['z']
    x_sym = semi_expl['x']
    p_sym = semi_expl['p']
    alg_expr = semi_expl['alg']

    # solve algebraically
    z_solved = ca.solve(ca.jacobian(alg_expr, z_sym), -ca.substitute(alg_expr, z_sym, ca.DM.zeros(z_sym.shape)))

    # substitute into ode rhs
    ode_rhs_expr = semi_expl['ode']
    ode_fn = ca.Function('ode_fn', [x_sym, p_sym], [ca.substitute(ode_rhs_expr, z_sym, z_solved)])

    return ode_fn