"""
backends/julia_backend.py
==========================
Julia backend for Braid DAE/ODE simulation.
Integrates ODE models using DifferentialEquations.jl and DiffEqGPU.jl.
"""

import os
import tempfile
import subprocess
import json
import numpy as np

class JuliaSimulationResult:
    def __init__(self, t_arr, y_arr, success=True):
        self.t = t_arr
        self.y = y_arr
        self.success = success


def _detect_differential_vars(ir: dict) -> list:
    """
    Returns a list of booleans indicating which states are differential (True)
    vs algebraic (False). A state is algebraic if its derivative (xdot) does
    not appear in the residuals.
    """
    xdot_names = ir.get('xdots', [])
    residuals = ir.get('residuals', [])
    
    # Track which xdots are found in the residuals
    found_xdots = set()
    
    def walk_ast(node):
        if not isinstance(node, dict):
            return
        if node.get('op') == 'var':
            name = node.get('name')
            if name in xdot_names:
                found_xdots.add(name)
        for arg in node.get('args', []):
            walk_ast(arg)
            
    for entry in residuals:
        walk_ast(entry)

        
    diff_vars = []
    for xdot in xdot_names:
        diff_vars.append(xdot in found_xdots)
        
    return diff_vars


def simulate_julia(dae_resolved, t_span, y0, params, method, **kwargs):
    """
    Simulates the system using Julia's DifferentialEquations.jl and DiffEqGPU.jl.
    """
    from lowering.julia_lowering import generate_julia_ode_code, generate_julia_dae_code
    
    model_type = dae_resolved.get('model_type', 'ODE')
    is_dae = (model_type == 'DAE')
    
    # ── 1. Determine Solver Method ───────────────────────────────────────────
    if method is None:
        method = "DABDF2()" if is_dae else "Tsit5()"
    
    # ── 2. Format Initial States and Parameters ──────────────────────────────
    y0_arr = np.array(y0, dtype=np.float64)
    params_arr = np.array(params, dtype=np.float64)
    
    is_batched = (y0_arr.ndim > 1) or (params_arr.ndim > 1)
    
    # Ensure batched sizes are consistent
    batch_size = 1
    if is_batched:
        if y0_arr.ndim > 1:
            batch_size = y0_arr.shape[0]
        if params_arr.ndim > 1:
            batch_size = params_arr.shape[0]
            
    # Generate the Julia function definition
    if is_dae:
        julia_func_code = generate_julia_dae_code(dae_resolved, func_name="f_dae")
        diff_vars = _detect_differential_vars(dae_resolved)
        diff_vars_str = "[" + ", ".join("true" if d else "false" for d in diff_vars) + "]"
        
        # yp0 initial derivative values
        yp0 = kwargs.get('yp0', None)
        if yp0 is None:
            yp0_arr = np.zeros_like(y0_arr)
        else:
            yp0_arr = np.array(yp0, dtype=np.float64)
    else:
        julia_func_code = generate_julia_ode_code(dae_resolved, func_name="f_ode")
    
    # Resolve t_eval times
    t_eval = kwargs.get('t_eval', None)
    if t_eval is not None:
        t_eval_str = "[" + ", ".join(map(str, t_eval)) + "]"
    else:
        num_steps = kwargs.get('num_steps', 100)
        t_eval_str = f"range({t_span[0]}, {t_span[1]}, length={num_steps})"
        
    # Write Julia script to run the simulation
    julia_script = []
    
    # Auto-dependency installer block
    julia_script.append("""
using Pkg
needed_pkgs = ["DifferentialEquations", "DiffEqBase", "JSON"]
for pkg in needed_pkgs
    if !haskey(Pkg.dependencies(), pkg)
        Pkg.add(pkg)
    end
end
using DifferentialEquations
using DiffEqBase
using JSON


# Retrieve BDF solvers from loaded modules
global DABDF2 = nothing
global DImplicitEuler = nothing
for (k, v) in Base.loaded_modules
    if Symbol(k.name) == :OrdinaryDiffEqBDF
        global DABDF2 = v.DABDF2
        global DImplicitEuler = v.DImplicitEuler
    end
end
""")


    if is_batched:
        julia_script.append("""
if !haskey(Pkg.dependencies(), "DiffEqGPU")
    Pkg.add("DiffEqGPU")
end
try
    using DiffEqGPU
    # If CUDA is available, import it
    if !haskey(Pkg.dependencies(), "CUDA")
        Pkg.add("CUDA")
    end
    using CUDA
catch e
    @warn "DiffEqGPU/CUDA import failed, falling back to CPU multi-threading." e
end
""")

    # Append the compiled function
    julia_script.append(julia_func_code)
    
    # Resolve tolerance options
    rtol = kwargs.get('rtol', None)
    atol = kwargs.get('atol', None)
    solve_opts = []
    if rtol is not None:
        solve_opts.append(f"reltol={rtol}")
    if atol is not None:
        solve_opts.append(f"abstol={atol}")
    if is_dae:
        solve_opts.append("initializealg=NoInit()")


    solve_opts_str = ", ".join(solve_opts)
    if solve_opts_str:
        solve_opts_str = ", " + solve_opts_str


    # Set up single vs batch ensemble problem
    if not is_batched:
        u0_float = [float(v) for v in y0_arr]
        p_float = [float(v) for v in params_arr]
        
        if is_dae:
            du0_float = [float(v) for v in yp0_arr]
            julia_script.append(f"""
u0 = {u0_float}
du0 = {du0_float}
p = {p_float}
tspan = ({t_span[0]}, {t_span[1]})
diff_vars = {diff_vars_str}
prob = DAEProblem(f_dae, du0, u0, tspan, p; differential_vars=diff_vars)
sol = solve(prob, {method}, saveat={t_eval_str}{solve_opts_str})

# Serialize output
out = Dict(
    "t" => sol.t,
    "y" => [sol[i, :] for i in 1:length(u0)]
)
print(JSON.json(out))
""")
        else:
            julia_script.append(f"""
u0 = {u0_float}
p = {p_float}
tspan = ({t_span[0]}, {t_span[1]})
prob = ODEProblem(f_ode, u0, tspan, p)
sol = solve(prob, {method}, saveat={t_eval_str}{solve_opts_str})

# Serialize output
out = Dict(
    "t" => sol.t,
    "y" => [sol[i, :] for i in 1:length(u0)]
)
print(JSON.json(out))
""")

    else:
        # Format batch data arrays
        u0_list = []
        if y0_arr.ndim > 1:
            for row in y0_arr:
                u0_list.append([float(val) for val in row])
        else:
            for _ in range(batch_size):
                u0_list.append([float(val) for val in y0_arr])
                
        p_list = []
        if params_arr.ndim > 1:
            for row in params_arr:
                p_list.append([float(val) for val in row])
        else:
            for _ in range(batch_size):
                p_list.append([float(val) for val in params_arr])
                
        if is_dae:
            du0_list = []
            if yp0_arr.ndim > 1:
                for row in yp0_arr:
                    du0_list.append([float(val) for val in row])
            else:
                for _ in range(batch_size):
                    du0_list.append([float(val) for val in yp0_arr])
            
            julia_script.append(f"""
u0_batch = {u0_list}
du0_batch = {du0_list}
p_batch = {p_list}
batch_size = {batch_size}
tspan = ({t_span[0]}, {t_span[1]})
diff_vars = {diff_vars_str}

# Define base problem
prob = DAEProblem(f_dae, du0_batch[1], u0_batch[1], tspan, p_batch[1]; differential_vars=diff_vars)

# Define ensemble problem for batching
prob_func = (prob, i, repeat) -> remake(prob, u0=u0_batch[i], du0=du0_batch[i], p=p_batch[i])
ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

# Auto-detect if CUDA/GPU integration should be used
use_gpu = false
if @isdefined(CUDA) && CUDA.functional()
    use_gpu = true
end

if use_gpu
    sol = solve(ensemble_prob, {method}, EnsembleGPUKernel(CUDA.CUDABackend()), trajectories=batch_size, saveat={t_eval_str}{solve_opts_str})
else
    sol = solve(ensemble_prob, {method}, EnsembleThreads(), trajectories=batch_size, saveat={t_eval_str}{solve_opts_str})
end

# Extract and serialize outputs
y_out = [ [sol[b][i, :] for i in 1:length(u0_batch[1])] for b in 1:batch_size ]
out = Dict(
    "t" => sol[1].t,
    "y" => y_out
)
print(JSON.json(out))
""")
        else:
            julia_script.append(f"""
u0_batch = {u0_list}
p_batch = {p_list}
batch_size = {batch_size}
tspan = ({t_span[0]}, {t_span[1]})

# Define base problem
prob = ODEProblem(f_ode, u0_batch[1], tspan, p_batch[1])

# Define ensemble problem for batching
prob_func = (prob, i, repeat) -> remake(prob, u0=u0_batch[i], p=p_batch[i])
ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)

# Auto-detect if CUDA/GPU integration should be used
use_gpu = false
if @isdefined(CUDA) && CUDA.functional()
    use_gpu = true
end

if use_gpu
    sol = solve(ensemble_prob, {method}, EnsembleGPUKernel(CUDA.CUDABackend()), trajectories=batch_size, saveat={t_eval_str}{solve_opts_str})
else
    sol = solve(ensemble_prob, {method}, EnsembleThreads(), trajectories=batch_size, saveat={t_eval_str}{solve_opts_str})
end


# Extract and serialize outputs
# sol has shape [batch_size][n_states, n_time]
y_out = [ [sol[b][i, :] for i in 1:length(u0_batch[1])] for b in 1:batch_size ]
out = Dict(
    "t" => sol[1].t,
    "y" => y_out
)
print(JSON.json(out))
""")

    # Write script to temporary file and execute Julia
    with tempfile.NamedTemporaryFile(suffix=".jl", delete=False, mode="w") as f:
        f.write("\n".join(julia_script))
        temp_path = f.name
        
    try:
        cmd = ["julia", temp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("--- JULIA RUN STDOUT ---")
            print(proc.stdout)
            print("--- JULIA RUN STDERR ---")
            print(proc.stderr)
            raise RuntimeError(f"Julia execution failed with exit code {proc.returncode}")
            
        result = json.loads(proc.stdout)
        t_arr = np.array(result["t"])
        y_arr = np.array(result["y"])
        return JuliaSimulationResult(t_arr, y_arr)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


