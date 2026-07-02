import os
import shutil
import tempfile
import subprocess
import ctypes
import numpy as np
import sympy as sp
from sympy import ccode

import sundials4py.core as core
import sundials4py.cvodes as cv

class SundialsSimulationResult:
    def __init__(self, t_arr, y_arr):
        self.t = t_arr
        self.y = y_arr
        self.success = True

def generate_c_rhs(dae, c_path):
    """Generates a C file containing the RHS function for the ODE system."""
    t = dae.t
    states = dae.states
    p_symbols = dae.params
    
    sub_dict = {}
    for i, state in enumerate(states):
        sub_dict[state] = sp.Symbol(f"y[{i}]")
    for j, param in enumerate(p_symbols):
        sub_dict[param] = sp.Symbol(f"p[{j}]")
        
    eq_strings = []
    for i, state in enumerate(states):
        state_deriv = sp.Derivative(state, t)
        if state_deriv in dae.ode_assignments:
            expr = dae.ode_assignments[state_deriv]
            expr_sub = expr.subs(sub_dict)
            expr_c = ccode(expr_sub)
            eq_strings.append(f"    ydot[{i}] = {expr_c};")
        else:
            raise ValueError(f"Derivative of state {state} is not defined in ode_assignments.")
            
    eq_code = "\n".join(eq_strings)
    
    c_code = f"""#include <math.h>

#ifdef _WIN32
__declspec(dllexport)
#endif
int f_ode(double t, const double *y, double *ydot, const double *p) {{
{eq_code}
    return 0;
}}
"""
    with open(c_path, "w", encoding="utf-8") as f:
        f.write(c_code)

def compile_c_rhs(c_path, dll_path):
    """Compiles the generated C file into a shared library."""
    # Check for gcc
    for gcc_name in ["gcc", "x86_64-w64-mingw32-gcc"]:
        gcc_path = shutil.which(gcc_name)
        if gcc_path:
            cmd = [gcc_path, "-shared", "-O3", "-o", dll_path, c_path]
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass
            
    # Check for clang
    clang_path = shutil.which("clang")
    if clang_path:
        cmd = [clang_path, "-shared", "-O3", "-o", dll_path, c_path]
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass
            
    # Check for cl.exe (MSVC)
    cl_path = shutil.which("cl")
    if cl_path:
        cmd = [cl_path, "/LD", "/O2", c_path, f"/Fe{dll_path}"]
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass
            
    return False

def simulate_c(dae, t_span, y0, params, method=None, compile_c=True, compiler_required=False, **kwargs):
    """Simulates the DAE/ODE system using C code generation and SUNDIALS."""
    t0, tf = t_span
    y0_arr = np.asarray(y0, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)
    
    # Check if we should use batch/ensemble simulation.
    # Currently we implement single simulation; batch sweeps can be run in a loop.
    is_batched = (y0_arr.ndim > 1) or (params_arr.ndim > 1)
    if is_batched:
        # Resolve dimensions
        if y0_arr.ndim == 1:
            batch_size = params_arr.shape[0] if params_arr.ndim > 1 else 1
            y0_arr = np.tile(y0_arr, (batch_size, 1))
        elif params_arr.ndim == 1:
            batch_size = y0_arr.shape[0]
            params_arr = np.tile(params_arr, (batch_size, 1))
        else:
            batch_size = y0_arr.shape[0]
            
        t_list = []
        y_list = []
        for i in range(batch_size):
            res = simulate_c_single(dae, t_span, y0_arr[i], params_arr[i], method, compile_c, compiler_required, **kwargs)
            t_list.append(res.t)
            y_list.append(res.y)
            
        return SundialsSimulationResult(t_list[0], np.stack(y_list, axis=0))
    else:
        return simulate_c_single(dae, t_span, y0_arr, params_arr, method, compile_c, compiler_required, **kwargs)

def simulate_c_single(dae, t_span, y0, params, method=None, compile_c=True, compiler_required=False, **kwargs):
    t0, tf = t_span
    n_states = len(y0)
    
    # Create temp directory for compilation if enabled
    compiled_lib = None
    if compile_c:
        temp_dir = tempfile.mkdtemp()
        c_path = os.path.join(temp_dir, "model.c")
        dll_path = os.path.join(temp_dir, "model.dll" if os.name == 'nt' else "model.so")
        
        try:
            generate_c_rhs(dae, c_path)
            success = compile_c_rhs(c_path, dll_path)
            if success:
                # Load compiled library
                compiled_lib = ctypes.CDLL(dll_path)
                compiled_lib.f_ode.argtypes = [
                    ctypes.c_double,
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double),
                    ctypes.POINTER(ctypes.c_double)
                ]
                compiled_lib.f_ode.restype = ctypes.c_int
            else:
                if compiler_required:
                    raise RuntimeError("No compatible C compiler (gcc, clang, cl.exe) found in PATH.")
                else:
                    print("Warning: C compiler not found in PATH. Falling back to lambdified Python callback.")
        except Exception as e:
            if compiler_required:
                raise RuntimeError(f"C Compilation failed: {e}")
            else:
                print(f"Warning: C Compilation failed ({e}). Falling back to lambdified Python callback.")
            compiled_lib = None
            
    # Setup rhs callback
    if compiled_lib is not None:
        def rhs_fn(t_val, y_vec, ydot_vec, user_data):
            y_ptr = core.N_VGetArrayPointer(y_vec)
            ydot_ptr = core.N_VGetArrayPointer(ydot_vec)
            
            y_c = y_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ydot_c = ydot_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            p_c = params.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            
            return compiled_lib.f_ode(t_val, y_c, ydot_c, p_c)
    else:
        # Fallback Python callback
        from simulation import lambdify_system
        numpy_rhs = lambdify_system(dae, 'numpy')
        def rhs_fn(t_val, y_vec, ydot_vec, user_data):
            y_arr = core.N_VGetArrayPointer(y_vec)
            ydot_arr = core.N_VGetArrayPointer(ydot_vec)
            res = numpy_rhs(t_val, y_arr, params)
            ydot_arr[:] = res
            return 0

    # Initialize SUNDIALS Context
    ret, ctx = core.SUNContext_Create(0)
    if ret != 0:
        raise RuntimeError("Failed to create SUNDIALS SUNContext")

    # Set up vector structures
    y = core.N_VNew_Serial(n_states, ctx)
    y_arr = core.N_VGetArrayPointer(y)
    y_arr[:] = y0

    # Setup CVODE solver
    lmm = cv.CV_BDF
    if method and method.lower() == 'adams':
        lmm = cv.CV_ADAMS

    cvode_view = cv.CVodeCreate(lmm, ctx)
    cvode_mem = cvode_view.get()

    # Initialize CVODE with solver callbacks
    cv.CVodeInit(cvode_mem, rhs_fn, float(t0), y)

    # Set tolerances
    rtol = kwargs.get('rtol', 1e-6)
    atol = kwargs.get('atol', 1e-6)
    cv.CVodeSStolerances(cvode_mem, float(rtol), float(atol))

    # Configure dense linear solver (mandatory for CVODE BDF)
    A = core.SUNDenseMatrix(n_states, n_states, ctx)
    LS = core.SUNLinSol_Dense(y, A, ctx)
    cv.CVodeSetLinearSolver(cvode_mem, LS, A)

    # Simulation loop
    num_steps = kwargs.get('num_steps', 1000)
    saveat_points = np.linspace(t0, tf, num_steps + 1)
    
    t_list = [float(t0)]
    y_list = [y_arr.copy()]

    for target_t in saveat_points[1:]:
        ret_code, actual_t = cv.CVode(cvode_mem, float(target_t), y, cv.CV_NORMAL)
        if ret_code < 0:
            raise RuntimeError(f"SUNDIALS CVODE integration failed with error code {ret_code} at t={actual_t}")
        t_list.append(actual_t)
        y_list.append(y_arr.copy())

    # Clean up temp directory if compilation was used
    if compile_c and os.path.exists(temp_dir):
        try:
            # We must unload/garbage collect compiled_lib before deleting
            del compiled_lib
            # Force garbage collection to release DLL handle on Windows
            import gc
            gc.collect()
            shutil.rmtree(temp_dir)
        except Exception:
            pass

    return SundialsSimulationResult(np.array(t_list), np.array(y_list).T)
