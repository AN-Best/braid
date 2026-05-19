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

Most physics simulators for RL either lock you into rigid-body dynamics (MuJoCo, Brax) or require you to manually derive equations of motion. Braid takes a different approach: you describe your system using an acausal component library, and the compiler handles the rest — DAE assembly, index reduction, and code generation to JAX.

The result is a differentiable, GPU-parallel ODE that can be vmapped over thousands of rollouts simultaneously.

---

## How It Works

```
Components (Spring, Mass, Damper)
    ↓ Node() connections
SymPy DAE assembly (SystemDAE)
    ↓ Order Reduction & Pantelides Algorithm [WIP]
Explicit ODE [WIP]
    ↓ sympy2jax / lambdify [Planned]
JAX function
    ↓ jax.vmap + Diffrax
GPU-parallel trajectories
```

---

## Example: Mass-Spring-Damper Assembly

```python
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from base import System, Node
from index_reduction import order_reduction_pass

# Define components
mass   = Mass('mass', m=2.0)
spring = Spring('spring', k=10.0)
damper = Damper('damper', c=0.2)
ground = Ground('ground')

# Assemble system
system = System([mass, spring, damper, ground])
Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])

# Convert to SymPy DAE and perform order reduction
dae = system.to_dae()
reduced_dae = order_reduction_pass(dae)

# [WIP] Pantelides index reduction and JAX compilation...
```

---

## Current State (Architecture Transition)

**Note:** The project is currently transitioning its core backend from `CasADi` to `SymPy` to better support high-index DAE structural analysis (e.g. Index-3 mechanical systems) via the Pantelides algorithm.

**What works:**

- 1D translational mechanical components: `Mass`, `Spring`, `Damper`, `Ground`, `Force` using native `sympy.Derivative` objects.
- Acausal port-based connections via `Node` with automatic force summation and position/velocity matching.
- Automatic DAE assembly from the component graph into a `SystemDAE` object.
- **Order Reduction:** Automatic lowering of 2nd-order ODE constraints into 1st-order systems.

**Known limitations / WIP:**

- Pantelides algorithm for structural index reduction is under active development.
- `compile.py` (CasADi → JAX via JAXADi) is currently deprecated and awaiting a SymPy-to-JAX rewrite.
- Only 1D translational mechanical domain implemented.
- No Gymnasium wrapper yet.

---

## Dependencies

```
sympy
numpy
jax (Planned)
diffrax (Planned)
```

---

## Project Structure

```
braid/
    components/
        linear_mechanical_1D.py   # Mass, Spring, Damper, Ground, Force
    base.py                       # System, Node, Component
    sym_dae.py                    # SymPy SystemDAE representation
    index_reduction.py            # Order reduction pass (Pantelides WIP)
    compile.py                    # [Deprecated] CasADi → JAX compiler
```

---

## Background

Braid is inspired by Modelica's acausal modeling paradigm and the SciML ecosystem's ModelingToolkit.jl. The core insight is that the component architecture — not the backend — is the primary contribution.
