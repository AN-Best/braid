import numpy as np
import scipy.integrate

def rk4_step_numpy(f, t, y, h):
    k1 = np.array(f(t, y))
    k2 = np.array(f(t + h/2.0, y + h/2.0 * k1))
    k3 = np.array(f(t + h/2.0, y + h/2.0 * k2))
    k4 = np.array(f(t + h, y + h * k3))
    return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

def backward_euler_step_numpy(f, jac_fn, t, y, h):
    from scipy.optimize import root
    def g(y_next):
        return y_next - y - h * np.array(f(t + h, y_next))
    def jac_g(y_next):
        return np.eye(len(y)) - h * np.array(jac_fn(t + h, y_next))
    sol = root(g, y, jac=jac_g, method='hybr')
    return sol.x

STEPPERS_NUMPY = {
    'euler': lambda f, t, y, h: y + h * np.array(f(t, y)),
    'rk4': rk4_step_numpy
}

class NumpyCustomResult:
    def __init__(self, t_arr, y_arr):
        self.t = t_arr
        self.y = y_arr.T
        self.success = True

def simulate_numpy(ode_func_raw, jac_func_raw, t_span, y0, params, method, **kwargs):
    if method is None:
        method = 'RK45'
        
    method_lower = method.lower()
    
    t0, tf = t_span
    
    def f(t_val, y_val):
        return ode_func_raw(t_val, y_val, params)
        
    if method_lower in STEPPERS_NUMPY or method_lower == 'backward_euler':
        num_steps = kwargs.get('num_steps', 1000)
        h = (tf - t0) / num_steps
        
        t_list = [t0]
        y_list = [np.array(y0, dtype=np.float64)]
        
        curr_t = t0
        curr_y = y_list[0]
        
        if method_lower == 'backward_euler':
            def jac_fn(t_val, y_val):
                return jac_func_raw(t_val, y_val, params)
            for _ in range(num_steps):
                curr_y = backward_euler_step_numpy(f, jac_fn, curr_t, curr_y, h)
                curr_t += h
                t_list.append(curr_t)
                y_list.append(curr_y.copy())
        else:
            step_fn = STEPPERS_NUMPY[method_lower]
            for _ in range(num_steps):
                curr_y = step_fn(f, curr_t, curr_y, h)
                curr_t += h
                t_list.append(curr_t)
                y_list.append(curr_y.copy())
                
        return NumpyCustomResult(np.array(t_list), np.array(y_list))
    else:
        def jac(t_val, y_val):
            return jac_func_raw(t_val, y_val, params)
            
        if method in ('Radau', 'BDF', 'LSODA'):
            sol = scipy.integrate.solve_ivp(f, t_span, y0, method=method, jac=jac, **kwargs)
        else:
            sol = scipy.integrate.solve_ivp(f, t_span, y0, method=method, **kwargs)
        return sol
