<img width="1024" height="1024" alt="Hexagonal braid logo on purple" src="https://github.com/user-attachments/assets/9c0c00c2-a4c6-49da-a85b-a1cf58d18091" />
# Braid

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
CasADi DAE assembly
    ↓ dae_reduce_index + dae_map_semi_expl
Explicit ODE
    ↓ JAXADi convert()
JAX function
    ↓ jax.vmap + Diffrax
GPU-parallel trajectories
```

---

## Example: Mass-Spring-Damper

```python
from braid.components.linear_mechanical_1D import Mass, Spring, Damper, Ground
from braid.system import System, Node
from braid.compile import Compile

from jaxadi import convert
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
from jax import vmap
import jax.numpy as jnp

# Define components
mass   = Mass('mass', m=2.0)
spring = Spring('spring', k=10.0)
damper = Damper('damper', c=0.2)
ground = Ground('ground')

# Assemble system
system = System([mass, spring, damper, ground])
Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2')])
Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])

# Compile to JAX
ode_fn = Compile(system)
jax_fn = convert(ode_fn, compile=True)

# Integrate
def vector_field(t, y, args):
    return jax_fn(y, args)[0][:, 0]

sol = diffeqsolve(
    ODETerm(vector_field), Tsit5(),
    t0=0.0, t1=10.0, dt0=0.01,
    y0=jnp.array([1.0, 0.0]),
    args=jnp.array([2.0, 10.0, 0.2]),
    saveat=SaveAt(ts=jnp.linspace(0, 10, 1000))
)

# GPU-parallel rollouts via vmap
batch_y0 = jnp.array([[1.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
results = vmap(lambda y0: diffeqsolve(..., y0=y0, ...).ys)(batch_y0)
# results.shape = (3, 1000, 2)
```

---

## Current State (MVP)

**What works:**

- 1D translational mechanical components: `Mass`, `Spring`, `Damper`, `Ground`, `Force`
- Acausal port-based connections via `Node` with automatic force summation and kinematic consistency
- Automatic DAE assembly from component graph
- DAE index reduction via CasADi (`dae_reduce_index`, `dae_map_semi_expl`)
- Symbolic algebraic elimination to produce explicit ODE
- JAX code generation via JAXADi
- GPU-parallel rollouts via `jax.vmap` + Diffrax

**Known limitations:**

- Only 1D translational mechanical domain implemented
- No Gymnasium wrapper yet
- Backend is tightly coupled to CasADi (abstraction layer planned)
- JAXADi has limitations on very large expression graphs

---

## Roadmap

- `Force` actuator component for RL control inputs
- Gymnasium environment wrapper
- Train a policy on mass-spring-damper (PPO via Stable Baselines or similar)
- Additional domains: rotational mechanical, electrical
- Backend abstraction layer (CasADi → ModelingToolkit.jl via `juliacall`)
- 2D planar multi-body systems

---

## Dependencies

```
casadi
jaxadi
jax
diffrax
```

---

## Project Structure

```
braid/
    components/
        linear_mechanical_1D.py   # Mass, Spring, Damper, Ground
    system.py                     # System, Node
    compile.py                    # Compile: DAE → JAX function
```

---

## Background

Braid is inspired by Modelica's acausal modeling paradigm and the SciML ecosystem's ModelingToolkit.jl. The core insight is that the component architecture — not the backend — is the primary contribution. CasADi handles DAE index reduction and symbolic algebra; JAXADi and Diffrax handle GPU execution.
