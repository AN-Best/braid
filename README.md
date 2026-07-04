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
CasADi DAE assembly (System/CasadiDAE)
    ↓ Pantelides Index Reduction Pass (Structural DAE to index-1 DAE)
Index-1 DAE (Active Equations)
    ↓ Tearing Pass (Symbolic solver)
Explicit solved assignments
    ↓ JSON IR Serialization (ir.py)
JSON Intermediate Representation (Version 2.0 syntax)
    ↓ Code Generation / Lowering (NumPy, PyTorch, Julia)
Numeric simulation functions (vmap / parallel batch execution)
```

---

## Features & Simulator Backends

Braid has a fully implemented compiler middle-end and a high-performance numerical simulation engine:

- **Pantelides Index Reduction Pass**: Employs structural DAE index reduction using maximum bipartite matching and directed alternating graph reachability traversal (backed by `networkx`). Differentiates constraints to reduce the DAE system to index-1.
- **Tearing Pass**: Symbolically solves the DAE active equations for all matched solved variables (algebraic unknowns and state derivatives).
- **JSON Intermediate Representation (IR)**: Serializes/deserializes compiled `CasadiDAE` structures using Braid's Version 2.0 AST syntax. It explicitly embeds metadata like `"model_type": "ODE"` to determine solver compatibility.
- **Numerical Simulation Backend (`simulation.py` and `backends/` folder)**:
  - **NumPy Backend (`backends/numpy_backend.py`)**: Integrates with SciPy's `scipy.integrate.solve_ivp` supporting standard adaptive-step solvers (e.g., `RK45`, `BDF`, `LSODA`, `Radau`, etc.).
  - **PyTorch Backend (`backends/torch_backend.py`)**: Optimized for GPU-parallel execution and batched rollouts. Integrates with `torchdiffeq.odeint` / `odeint_adjoint` for parallel, differentiable ODE simulations.
  - **Julia Backend (`backends/julia_backend.py`)**: High-performance ODE integration via [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/). It generates allocation-free Julia functions `f(du, u, p, t)` at runtime, supporting multi-threaded CPU execution (`EnsembleThreads`) or NVIDIA GPUs (`EnsembleGPUKernel` via `DiffEqGPU.jl` + `CUDA.jl`) when available.
  - **GPU Acceleration & Batch Parallelization**: Supports batched initial conditions `y0` of shape `(batch_size, num_states)` and batched parameters `params` of shape `(batch_size, num_params)`. This allows running thousands of parallel simulations simultaneously on the GPU in a single vectorized pass (PyTorch backend) or via Julia's ensemble infrastructure.

---

## Installation & Setup

You can set up your environment using either **pip** or **conda**:

### Using Conda (Recommended for GPU/PyTorch/Julia users)
```bash
conda env create -f environment.yml
conda activate braid
```

### Using Pip
```bash
pip install -r requirements.txt
```

---

## Example: Compilation and Simulation

Here is how to compile a component system into an explicit ODE and simulate it:

```python
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from base import System, Node
from index_reduction import pantelides_pass, tearing_pass
from simulation import simulate_system
import numpy as np

# 1. Define components
mass   = Mass('mass', m=2.0)
spring = Spring('spring', k=10.0)
damper = Damper('damper', c=0.2)
ground = Ground('ground')

# 2. Assemble system
system = System([mass, spring, damper, ground])
Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])

# 3. Convert to DAE
dae = system.to_dae()

# 4. Perform compiler passes
index_reduced_dae = pantelides_pass(dae)
torn_dae = tearing_pass(index_reduced_dae)

# 5. Simulate single run on CPU (SciPy RK45)
t_span = (0.0, 5.0)
y0 = [1.0, 0.0]  # x_mass, v_mass
sol_cpu = simulate_system(torn_dae, t_span, y0, params=None, backend='numpy', method='RK45')

# 6. Simulate parallel batch on GPU (PyTorch/torchdiffeq)
batch_size = 1000
y0_batch = np.zeros((batch_size, 2))
y0_batch[:, 0] = np.linspace(0.5, 2.0, batch_size)  # varying initial stretch
sol_gpu = simulate_system(
    torn_dae, t_span, y0_batch, params=None,
    backend='pytorch', method='dopri5', device='cuda'
)

# 7. Simulate using Julia backend (DifferentialEquations.jl / DiffEqGPU.jl)
sol_julia = simulate_system(
    torn_dae, t_span, y0_batch, params=None,
    backend='julia', method='Tsit5()', num_steps=100
)
```

---

## Project Structure

```
braid/
    components/
        linear_mechanical_1D.py   # Mass, Spring, Damper, Ground, Force components
        electrical_basic.py       # Resistor, Capacitor, Ground, VoltageSource, Sensors
    base.py                       # System, Node, Component base classes
    casadi_dae.py                 # CasADi CasadiDAE representation
    index_reduction.py            # Pantelides and Tearing passes
    simulation.py                 # Core simulation wrapper and validation checks
    backends/                     # Modular backend implementations
        numpy_backend.py          # NumPy/SciPy solver integration
        torch_backend.py          # PyTorch/torchdiffeq solver integration
        julia_backend.py          # Julia/DifferentialEquations.jl & DiffEqGPU.jl bridge
    ir.py                         # JSON IR compiler and serializer (V2.0 syntax)
    lowering/                     # Code lowerings (CasADi C, PyTorch, Julia, NumPy)
    requirements.txt              # Standard Python pip requirements
    environment.yml               # Conda environment definition
    test/
        test_system_assembly.py   # Verification of acausal component compilation
        test_electrical_basic.py  # Verification of electrical systems
        test_julia_backend.py     # Verification of Julia simulation physics
```

---

## Background

Braid is inspired by Modelica's acausal modeling paradigm and the SciML ecosystem's ModelingToolkit.jl. The core insight is that the component architecture — not the backend — is the primary contribution.
