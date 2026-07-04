import numpy as np
import scipy.integrate

class NumpyCustomResult:
    def __init__(self, t_arr, y_arr, success=True):
        self.t = t_arr
        self.y = y_arr
        self.success = success

def simulate_numpy(ode_func_raw, jac_func_raw, t_span, y0, params, method, **kwargs):
    if method is None:
        method = 'RK45'
        
    def f(t_val, y_val):
        return ode_func_raw(t_val, y_val, params)
        
    def jac(t_val, y_val):
        return jac_func_raw(t_val, y_val, params)
        
    if method in ('Radau', 'BDF', 'LSODA'):
        sol = scipy.integrate.solve_ivp(f, t_span, y0, method=method, jac=jac, **kwargs)
    else:
        sol = scipy.integrate.solve_ivp(f, t_span, y0, method=method, **kwargs)
        
    return sol


def simulate_numpy_dae(residual_func, t_span, y0, yp0, params, method, **kwargs):
    from solve_dae.integrate import solve_dae
    if method is None:
        method = 'Radau'

    # solve_dae expects F(t, y, yp) -> residuals
    def res(t_val, y_val, yp_val):
        return residual_func(t_val, y_val, yp_val, params)

    sol = solve_dae(res, t_span, y0, yp0, method=method, **kwargs)
    return sol

