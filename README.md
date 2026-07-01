<img width="256" height="256" alt="Hexagonal braid logo on purple" src="https://github.com/user-attachments/assets/9c0c00c2-a4c6-49da-a85b-a1cf58d18091" /> </br>

# braid

**GPU-parallel acausal component-based simulation for reinforcement learning and optimal control.**

Braid lets you build physical systems from reusable components — springs, masses, dampers — connect them together, and automatically produce GPU-parallelizable simulations suitable for RL training and optimal control.

---

## Licensing

Braid is free for non-commercial use, including academic research, personal projects, and educational purposes.

**Commercial use requires a license.** If you are using Braid in an industry context — including research funded by industry, product development, or internal tooling — please get in touch to discuss licensing options.

📬 [Contact for commercial licensing](mailto:anbest.37.7@gmail.com)

---

## Motivation

Most physics simulators for RL either lock you into rigid-body dynamics (MuJoCo, Brax) or require you to manually derive equations of motion. Braid takes a different approach: you describe your system using an acausal component library, and the compiler handles the rest — DAE assembly, index reduction, tearing/elimination, and code generation to target numerical backends.

The result is a differentiable, GPU-parallel ODE that can be evaluated over thousands of rollouts/initial conditions simultaneously.

---

## How It Works

```
Components (Spring, Mass, Damper)
    ↓ Node() connections
SymPy DAE assembly (SystemDAE)
    ↓ Pantelides Index Reduction Pass (Structural DAE to index-1 DAE)
Index-1 DAE (Active Equations)
    ↓ Tearing Pass (Symbolic solver)
Explicit solved assignments
    ↓ Elimination & Symbolic Simplification Pass
Minimal ODE assignments (ode_assignments)
    ↓ JSON IR Serialization (json_ir.py)
JSON Intermediate Representation (srepr-encoded)
    ↓ Code Generation (NumPy, PyTorch, JAX, etc.)
Numeric simulation functions (vmap / parallel batch execution)
```

---

## Features & Simulator Backends

Braid has a fully implemented compiler middle-end and a high-performance numerical simulation engine:

- **Pantelides Index Reduction Pass**: Employs structural DAE index reduction using maximum bipartite matching and directed alternating graph reachability traversal (backed by `networkx`). Differentiates constraints to reduce the DAE system to index-1.
- **Tearing Pass**: Symbolically solves the DAE active equations for all matched solved variables (algebraic unknowns and state derivatives).
- **Elimination & Symbolic Simplification Pass**:
  - Automatically identifies state derivative definitions of the form `Derivative(A, t) = B` to eliminate redundant states.
  - Generates a minimal state derivative mapping (`ode_assignments`) for numerical ODE integration.
  - Symbols are simplified using SymPy's `sp.simplify`.
- **JSON Intermediate Representation (IR)**: Serializes/deserializes compiled `SystemDAE` structures using SymPy's `srepr` encoding to preserve exact symbolic representations across languages.
- **Numerical Simulation Backend (`simulation.py` and `backends/` folder)**:
  - **NumPy Backend (`backends/numpy_backend.py`)**: Designed for CPU-based simulation.
    - *SciPy Integrator*: Uses SciPy's `scipy.integrate.solve_ivp` to support standard adaptive-step solvers (e.g., `RK45`, `BDF`, `LSODA`, `Radau`, etc.). The analytical Jacobian derived symbolically is automatically supplied to stiff solvers (`Radau`, `BDF`, `LSODA`) to speed up integration.
    - *Custom Fixed-Step Solvers*:
      - Forward Euler (`euler`)
      - Runge-Kutta 4th Order (`rk4`)
      - Implicit Backward Euler (`backward_euler` utilizing SciPy's `scipy.optimize.root` solver with the analytical Jacobian for accelerated root-finding)
  - **PyTorch Backend (`backends/torch_backend.py`)**: Optimized for GPU-parallel execution and batched rollouts.
    - *Custom Fixed-Step Solvers*:
      - Forward Euler (`euler`)
      - Runge-Kutta 4th Order (`rk4`)
      - Implicit Backward Euler (`backward_euler` using a custom on-device Newton-Raphson solver that evaluates the symbolically compiled analytical Jacobian, eliminating PyTorch autograd graph-tracing overhead and improving performance by ~30%)
  - **Julia Backend (`backends/julia_backend.py`)**: High-performance ODE integration via [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/), bridged into Python through `juliacall`. Braid compiles the symbolic `ode_assignments` directly into a Julia `ODEProblem` at runtime using SymPy's Julia code printer.
    - *Adaptive Solvers*:
      - `Tsit5` — explicit Runge-Kutta (4/5 order), excellent general-purpose adaptive solver
      - `Rodas4` — stiff Rosenbrock solver; ideal for stiff mechanical/electrical systems
      - `Radau` — implicit Runge-Kutta for highly stiff problems
      - `BDF` — multi-step implicit solver; good for index-1 DAE residuals after reduction
    - *Fixed-Step Solvers*:
      - `RK4` — classic 4th-order Runge-Kutta
      - `ImplicitEuler` — first-order A-stable implicit method
      - `Euler` — explicit forward Euler
    - *Batched Ensemble Simulation*: Supports batched `y0` and `params` via Julia's `EnsembleProblem` API, dispatching across CPU threads (`EnsembleThreads`) or NVIDIA GPUs (`EnsembleGPUArray` via `DiffEqGPU.jl` + `CUDA.jl`) when available.
    - *Subprocess Isolation*: Because `juliacall` holds the Python GIL during execution, the Julia worker runs in a dedicated child process (`gui/julia_worker.py`) that communicates with the FastAPI server over a newline-delimited JSON protocol, keeping the async event loop fully responsive.
  - **GPU Acceleration & Batch Parallelization**: Supports batched initial conditions `y0` of shape `(batch_size, num_states)` and batched parameters `params` of shape `(batch_size, num_params)`. This allows running thousands of parallel simulations simultaneously on the GPU in a single vectorized pass (PyTorch backend) or via Julia's ensemble infrastructure.

---

## Installation & Setup

You can set up your environment using either **pip** or **conda**:

### Using Conda (Recommended for GPU/PyTorch users)
```bash
conda env create -f environment.yml
conda activate braid
```

### Using Pip
```bash
pip install -r requirements.txt
```

---

## Example: Compilation and Serialization

Here is how to compile a component system into an explicit ODE and serialize it to JSON:

```python
import sympy as sp
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from base import System, Node
from index_reduction import (
    pantelides_pass,
    tearing_pass,
    simplification_pass
)
from json_ir import to_json

# 1. Define components
mass   = Mass('mass', m=2.0)
spring = Spring('spring', k=10.0)
damper = Damper('damper', c=0.2)
ground = Ground('ground')

# 2. Assemble system
system = System([mass, spring, damper, ground])
Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])

# 3. Convert to SymPy DAE
dae = system.to_dae()

# 4. Perform compiler passes
index_reduced_dae = pantelides_pass(dae)
torn_dae = tearing_pass(index_reduced_dae)
simplified_dae = simplification_pass(torn_dae)

# 5. Serialize to JSON IR
json_str = to_json(simplified_dae)
with open("mass_spring_damper_ode.json", "w") as f:
    f.write(json_str)

print("System compiled and saved successfully!")
```

---

## Example: Simulating the Compiled System (CPU, GPU, or Julia)

You can easily integrate your compiled system using the simulation engine:

```python
from simulation import simulate_system
import numpy as np

# Let's say you compiled the pendulum DAE to `simplified_dae`
t_span = (0.0, 1.5)

# --- 1. Single CPU simulation (using NumPy / SciPy RK45) ---
y0 = [1.0, 0.0, 0.0, 0.0, 0.0]       # initial states
params_val = [1.0, 9.81, 1.0]        # m, g, L
sol = simulate_system(simplified_dae, t_span, y0, params_val, backend='numpy', method='RK45')

# --- 2. Parallel batched GPU simulation (using PyTorch) ---
batch_size = 1000
L_vals = np.linspace(0.5, 2.0, batch_size)

# y0 batch: shape (batch_size, num_states)
y0_batch = np.zeros((batch_size, len(simplified_dae.states)))
y0_batch[:, 0] = L_vals  # x starts at L

# params batch: shape (batch_size, num_params)
params_batch = np.zeros((batch_size, 3))
params_batch[:, 0] = 1.0     # m = 1.0
params_batch[:, 1] = 9.81    # g = 9.81
params_batch[:, 2] = L_vals  # L

# Simulate 1000 pendulums simultaneously on the GPU!
sol_batch = simulate_system(
    simplified_dae, t_span, y0_batch, params_batch,
    backend='pytorch', method='rk4', device='cuda', num_steps=1000
)
# Resulting y shape: (batch_size, num_states, num_steps + 1)
print("Parallel simulation shape:", sol_batch.y.shape)

# --- 3. Julia backend (DifferentialEquations.jl) ---
# Adaptive stiff solver — great for mechanically stiff systems
from backends.julia_backend import simulate_julia
sol_julia = simulate_julia(
    simplified_dae, t_span, y0, params_val,
    method='tsit5',       # or 'rodas4', 'radau', 'bdf', 'rk4', ...
    num_steps=1000
)
print("Julia result shape:", sol_julia.y.shape)  # (num_states, num_steps+1)

# Batched Julia ensemble (CPU threads or NVIDIA GPU via DiffEqGPU.jl)
sol_ensemble = simulate_julia(
    simplified_dae, t_span, y0_batch, params_batch,
    method='tsit5', device='cuda', num_steps=1000
)
print("Ensemble shape:", sol_ensemble.y.shape)  # (batch_size, num_states, num_steps+1)
```

### Julia Solver Reference

| Method key | DifferentialEquations.jl solver | Notes |
|---|---|---|
| `tsit5` | `Tsit5()` | Adaptive RK (4/5); best general-purpose choice |
| `rodas4` | `Rodas4()` | Stiff Rosenbrock; recommended for stiff mechanical/electrical systems |
| `radau` | `Radau()` | Implicit RK; robust for highly stiff problems |
| `bdf` | `BDF()` | Multi-step implicit; good for index-1 residuals |
| `rk4` | `RK4()` | Fixed-step classical RK4 |
| `backward_euler` | `ImplicitEuler()` | Fixed-step first-order implicit |
| `euler` | `Euler()` | Fixed-step forward Euler |

---

## Project Structure

```
braid/
    components/
        linear_mechanical_1D.py   # Mass, Spring, Damper, Ground, Force, Sensor components
        metadata.json             # Component registry for the GUI palette
    base.py                       # System, Node, Component base classes
    sym_dae.py                    # SymPy SystemDAE representation & metadata
    index_reduction.py            # Pantelides, Tearing, and Simplification passes
    simulation.py                 # Core simulation wrapper and compiler (lambdify)
    backends/                     # Modular backend implementations
        numpy_backend.py          # NumPy/SciPy solver and custom stepper implementation
        torch_backend.py          # PyTorch GPU-parallel solver and custom stepper implementation
        julia_backend.py          # Julia/DifferentialEquations.jl bridge via juliacall
    json_ir.py                    # JSON IR serializer/deserializer using srepr
    gui/
        app.py                    # FastAPI server — compile & simulate REST endpoints
        julia_worker.py           # Julia subprocess worker (JSON stdio protocol)
        templates/
            index.html            # Interactive browser GUI (vis-network + Plotly)
    requirements.txt              # Standard Python pip requirements
    environment.yml               # Conda environment definition
    test/
        test_pantelides.py        # Verification of compiler passes (Pendulum & Mass-Spring-Damper)
        test_system_assembly.py   # Verification of acausal component compilation
        test_simulation.py        # Verification of simulation solvers and GPU batching
```

---

## Interactive GUI

Braid ships with a browser-based graphical interface for building, compiling, and simulating physical systems without writing any code.

```bash
conda activate braid
cd gui
python app.py
```

Then open **http://127.0.0.1:8000** in your browser.

**Features:**
- Drag-and-drop component palette with live parameter editing
- Visual acausal graph canvas (vis-network)
- One-click compilation with a 4-stage equation pipeline viewer (DAE → Index-Reduced → Torn → ODE)
- Simulation with all three backends (NumPy, PyTorch, Julia) selectable from a dropdown
- Live Julia status badge that polls `/api/julia/status` and shows initializing / ready / error states
- Plotly trajectory plots with sensor filtering
- Save/load model layouts as `.braid` JSON files

> **Julia in the GUI:** On first use, the Julia worker subprocess initializes `DifferentialEquations.jl` in the background (typically 1–3 minutes). The status badge in the Simulator Config panel turns green once ready. The worker process is kept alive for the lifetime of the server, so subsequent Julia simulations have negligible startup overhead.

---

## Background

Braid is inspired by Modelica's acausal modeling paradigm and the SciML ecosystem's ModelingToolkit.jl. The core insight is that the component architecture — not the backend — is the primary contribution.
