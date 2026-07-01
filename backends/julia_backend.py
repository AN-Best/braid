import os
# pyrefly: ignore [missing-import]
import numpy as np
import sympy as sp
from sympy.printing.julia import julia_code

# Defer import to avoid boot time issues if julia is not used.
jl = None
Main = None
_julia_initialized = False
JULIA_INIT_ERROR = None   # Stores any error message from background initialization
_julia_cuda_available = False  # Whether CUDA loaded successfully

def init_julia():
    global _julia_initialized, jl, Main, JULIA_INIT_ERROR, _julia_cuda_available
    if _julia_initialized:
        return

    print("Initializing Julia environment and dependencies (this might take a moment)...")
    # Configure environment to disable compiled modules to bypass Windows Application
    # Control policy DLL blocking.
    os.environ["PYTHON_JULIACALL_COMPILED_MODULES"] = "no"
    os.environ["JULIAOPTS"] = "--compiled-modules=no"
    os.environ["JULIA_PKG_PRECOMPILE_AUTO"] = "0"

    import juliapkg
    # Core ODE packages — always required.
    juliapkg.add("DifferentialEquations", "0c46a032-eb83-5123-abaf-570d42b7fbaa")
    juliapkg.add("OrdinaryDiffEq", "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed")
    juliapkg.add("OrdinaryDiffEqLowOrderRK")
    juliapkg.add("OrdinaryDiffEqSDIRK")
    juliapkg.add("StaticArrays", "90137ffa-7385-5640-81b9-e52037218182")
    # GPU packages — optional, only added when CUDA is available.
    try:
        juliapkg.add("DiffEqGPU", "071ae1c0-96b5-11e9-1965-c90190d839ea")
        juliapkg.add("CUDA", "052768ef-5323-5732-b1bb-66c8b64840ba")
    except Exception:
        pass
    juliapkg.resolve()

    from juliacall import Main as julia_main
    jl = julia_main

    # Core packages
    jl.seval("using DifferentialEquations")
    jl.seval("using OrdinaryDiffEq")
    jl.seval("using OrdinaryDiffEqLowOrderRK")
    jl.seval("using OrdinaryDiffEqSDIRK")
    jl.seval("using StaticArrays")

    # GPU packages — optional
    try:
        jl.seval("using DiffEqGPU")
        jl.seval("using CUDA")
        _julia_cuda_available = bool(jl.seval("CUDA.functional()"))
        print(f"Julia CUDA available: {_julia_cuda_available}")
    except Exception as cuda_err:
        print(f"Julia CUDA not available (GPU ensemble disabled): {cuda_err}")
        _julia_cuda_available = False

    _julia_initialized = True
    print("Julia initialization complete.")

class JuliaSimulationResult:
    def __init__(self, t_arr, y_arr):
        self.t = t_arr
        self.y = y_arr
        self.success = True

# Method mappings from Braid method names to DifferentialEquations.jl solvers
METHODS_MAP = {
    'euler': 'Euler()',
    'rk4': 'RK4()',
    'backward_euler': 'ImplicitEuler()',
    'tsit5': 'Tsit5()',
    'rodas4': 'Rodas4()',
    'radau': 'Radau()',
    'bdf': 'BDF()',
    'rk45': 'Tsit5()'
}

def simulate_julia(dae, t_span, y0, params, method=None, device=None, **kwargs):
    init_julia()
    
    t0, tf = t_span
    num_steps = kwargs.get('num_steps', 1000)
    h = (tf - t0) / num_steps
    
    # Parse method
    method_lower = (method or 'rk4').lower()
    julia_alg = METHODS_MAP.get(method_lower, 'Tsit5()')
    
    # Check if y0 and params are batched
    y0_arr = np.asarray(y0, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)
    
    is_batched = (y0_arr.ndim > 1) or (params_arr.ndim > 1)
    
    # Ensure both are aligned batched if either is batched
    if is_batched:
        if y0_arr.ndim == 1:
            batch_size = params_arr.shape[0] if params_arr.ndim > 1 else 1
            y0_arr = np.tile(y0_arr, (batch_size, 1))
        elif params_arr.ndim == 1:
            batch_size = y0_arr.shape[0]
            params_arr = np.tile(params_arr, (batch_size, 1))
        else:
            batch_size = y0_arr.shape[0]
    else:
        batch_size = 1

    # Generate Julia ODE function
    t = dae.t
    states = dae.states
    p_symbols = dae.params
    
    sub_dict = {}
    for i, state in enumerate(states):
        sub_dict[state] = sp.Symbol(f"u[{i+1}]")
    for j, param in enumerate(p_symbols):
        sub_dict[param] = sp.Symbol(f"p[{j+1}]")
        
    eq_strings = []
    for state in states:
        state_deriv = sp.Derivative(state, t)
        if state_deriv in dae.ode_assignments:
            expr = dae.ode_assignments[state_deriv]
            expr_julia = julia_code(expr.subs(sub_dict))
            eq_strings.append(expr_julia)
        else:
            raise ValueError(f"Derivative of state {state} is not defined in ode_assignments.")
            
    # Define ODE function in Julia
    if len(states) <= 100:
        julia_fn_str = f"""
        function f_ode(u, p, t)
            return SA[{', '.join(eq_strings)}]
        end
        """
    else:
        julia_fn_str = f"""
        function f_ode(u, p, t)
            return [{', '.join(eq_strings)}]
        end
        """
    jl.seval(julia_fn_str)
    
    # Save points grid
    saveat_points = np.linspace(t0, tf, num_steps + 1)
    
    # Setup solve kwargs
    solve_kwargs = f"saveat={[float(x) for x in saveat_points]}"
    if method_lower in ('euler', 'rk4', 'backward_euler'):
        # Fixed step methods require dt and adaptive=false
        solve_kwargs += f", dt={float(h)}, adaptive=false"
        
    if not is_batched:
        # Single run
        jl.y0 = list(y0_arr)
        jl.p = list(params_arr)
        jl.tspan = (float(t0), float(tf))
        
        jl.seval("prob = ODEProblem(f_ode, SVector{" + str(len(y0_arr)) + "}(Float64[y0...]), tspan, Float64[p...])")
        jl.seval(f"sol = solve(prob, {julia_alg}, {solve_kwargs})")
        
        # Extract results
        sol_t = np.array(jl.seval("sol.t"))
        sol_u = np.array(jl.seval("reduce(hcat, sol.u)"))
        return JuliaSimulationResult(sol_t, sol_u)
    else:
        # Batched run (EnsembleProblem)
        jl.y0_batch = [list(row) for row in y0_arr]
        jl.params_batch = [list(row) for row in params_arr]
        jl.tspan = (float(t0), float(tf))
        
        num_states = len(y0_arr[0])
        
        # Convert batch arrays to native Julia typed arrays to avoid GIL issues in parallel threads
        jl.seval("y0_batch_jl = [SVector{" + str(num_states) + ", Float64}(Float64[row...]) for row in y0_batch]")
        jl.seval("params_batch_jl = [Float64[row...] for row in params_batch]")
        
        jl.seval("base_prob = ODEProblem(f_ode, y0_batch_jl[1], tspan, params_batch_jl[1])")
        
        # Define prob_func using modern SciMLBase EnsembleContext interface
        jl.seval("function prob_func(prob, ctx)\n"
                 "    remake(prob, u0=y0_batch_jl[ctx.sim_id], p=params_batch_jl[ctx.sim_id])\n"
                 "end")
        
        jl.seval("ensemble_prob = EnsembleProblem(base_prob, prob_func=prob_func)")
        
        # Decide solver ensemble backend (GPU vs CPU threads)
        if device is None:
            jl_device = 'cuda' if _julia_cuda_available else 'cpu'
        else:
            jl_device = str(device).lower()

        if jl_device == 'cuda':
            if not _julia_cuda_available:
                print("Warning: CUDA device requested, but CUDA is not functional in Julia. Falling back to CPU threads.")
                ensemble_backend = "EnsembleThreads()"
            else:
                ensemble_backend = "EnsembleGPUArray(CUDA.CUDABackend())"
        else:
            ensemble_backend = "EnsembleThreads()"
            
        jl.seval(f"sol = solve(ensemble_prob, {julia_alg}, {ensemble_backend}, trajectories={batch_size}, {solve_kwargs})")
        
        # Extract ensemble solution
        sol_list = []
        for i in range(batch_size):
            traj_u = np.array(jl.seval(f"reduce(hcat, sol.u[{i+1}].u)"))
            sol_list.append(traj_u)
            
        y_out = np.stack(sol_list, axis=0)
        sol_t = np.array(jl.seval("sol.u[1].t"))
        
        return JuliaSimulationResult(sol_t, y_out)
