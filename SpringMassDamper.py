from base import System, Node
from components.linear_mechanical_1D import Mass, Spring, Damper, Ground, Force
from compile import Compile
from jaxadi import convert
from backend_conversion import casadi2jax
import numpy as np
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt
import matplotlib.pyplot as plt
from jax import vmap


mass = Mass('mass',m = 1.0)
spring = Spring('spring', k = 1.0)
damper = Damper('damper', c = 0.5)
ground = Ground('ground')
force = Force('force', F = 100.0)

system = System([mass,spring,damper,ground,force])
Node(system, [(mass, 'p'), (spring, 'p2'), (damper, 'p2'),(force,'p')])
Node(system, [(ground, 'p'), (spring, 'p1'), (damper, 'p1')])

ode_fn = Compile(system=system)
p = System.get_param_vector(system)
ode_fn_expanded = ode_fn.expand()
jax_fn = casadi2jax(ode_fn_expanded)

def vector_field(t, y, args):
    return jax_fn(y, args)[0][:, 0]

term = ODETerm(vector_field)
solver = Tsit5()
t0, t1, dt0 = 0.0, 30.0, 0.01
y0 = jnp.array([1.0, 0.0])
p = jnp.array(p)
saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))

sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=p, saveat=saveat)

t = np.array(sol.ts)
x = np.array(sol.ys[:, 0])
xdot = np.array(sol.ys[:, 1])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Mass Position')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, xdot)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Mass Velocity')
plt.grid(True)

plt.tight_layout()
plt.show()

batch_y0 = jnp.array([[1.0, 0.0], [0.5, 0.0], [2.0, 0.0], [1.0, 1.0]])

def single_rollout(y0):
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args=p, saveat=saveat)
    return sol.ys

batch_rollout = vmap(single_rollout)
results = batch_rollout(batch_y0)
print(results.shape)