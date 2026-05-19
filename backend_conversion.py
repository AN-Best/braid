from jaxadi import convert

def casadi2jax(caODE):
    jax_fn = convert(caODE, compile=True)
    return jax_fn




