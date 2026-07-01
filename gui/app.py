import sys
import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root directory to sys.path to allow imports from sibling folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pkgutil
import importlib
import inspect
from base import System, Node, Component
from index_reduction import pantelides_pass, tearing_pass, simplification_pass
from simulation import simulate_system
from json_ir import to_json as dae_to_json

# --- Julia subprocess worker state ---
# juliacall holds the GIL during execution, so we run Julia in a child process
# instead of a thread to avoid deadlocking the uvicorn event loop.
#
# States: 'idle' | 'initializing' | 'ready' | 'error'
JULIA_STATUS = "idle"
JULIA_INIT_ERROR_MSG: Optional[str] = None
_julia_proc: Optional[asyncio.subprocess.Process] = None
_julia_lock = asyncio.Lock()  # Serializes access to the worker subprocess
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "julia_worker.py")


app = FastAPI(title="Braid GUI API Server")

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for compiled DAE models
COMPILED_MODELS: Dict[str, Any] = {}

# Load components metadata from metadata.json
METADATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "components", "metadata.json")

def load_components_metadata() -> Dict[str, Any]:
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading component metadata: {e}")
        return {}

COMPONENTS = load_components_metadata()

# Dynamic component class auto-discovery
def get_component_classes() -> Dict[str, Any]:
    comp_classes = {}
    try:
        import components
        for _, module_name, is_pkg in pkgutil.walk_packages(components.__path__, components.__name__ + '.'):
            if not is_pkg:
                try:
                    mod = importlib.import_module(module_name)
                    for name, obj in inspect.getmembers(mod, inspect.isclass):
                        if issubclass(obj, Component) and obj is not Component:
                            comp_classes[name] = obj
                except Exception as e:
                    print(f"Error loading module {module_name}: {e}")
    except Exception as e:
        print(f"Error walking components package: {e}")
    return comp_classes


# --- Pydantic Request Models ---

class NodeSchema(BaseModel):
    id: str
    type: str
    params: Dict[str, float]

class EdgeSchema(BaseModel):
    source: str
    sourcePort: str
    target: str
    targetPort: str

class CompileRequest(BaseModel):
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]

class SimulateRequest(BaseModel):
    compile_id: str
    t_span: List[float]
    y0: Dict[str, float]
    params: Dict[str, float]
    backend: str
    method: Optional[str] = None
    num_steps: int = 1000

# --- Helper Functions ---

def build_connected_groups(nodes: List[NodeSchema], edges: List[EdgeSchema]) -> List[List[tuple]]:
    """Finds all connected components of ports to group them into Node junctions."""
    from collections import defaultdict
    
    # 1. Map component port ids to check validity
    valid_ports = set()
    for n in nodes:
        if n.type in COMPONENTS:
            for p in COMPONENTS[n.type]["ports"]:
                valid_ports.add((n.id, p))
                
    # 2. Build adjacency list of connections
    adj = defaultdict(list)
    for e in edges:
        u = (e.source, e.sourcePort)
        v = (e.target, e.targetPort)
        if u in valid_ports and v in valid_ports:
            adj[u].append(v)
            adj[v].append(u)
            
    # 3. Find components via BFS
    visited = set()
    groups = []
    
    for port in list(adj.keys()):
        if port not in visited:
            comp = []
            queue = [port]
            visited.add(port)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            groups.append(comp)
            
    return groups

# --- Julia subprocess management ---

async def _start_julia_worker():
    """Launch the Julia worker subprocess and wait for its ready handshake.
    Must be called while holding _julia_lock."""
    global JULIA_STATUS, JULIA_INIT_ERROR_MSG, _julia_proc

    if JULIA_STATUS in ("ready", "initializing"):
        return

    JULIA_STATUS = "initializing"
    print("Starting Julia worker subprocess...")

    python_exe = sys.executable
    _julia_proc = await asyncio.create_subprocess_exec(
        python_exe, _WORKER_SCRIPT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Read the ready handshake line (Julia init can take minutes)
    try:
        raw = await _julia_proc.stdout.readline()
        msg = json.loads(raw.decode().strip())
    except Exception as exc:
        JULIA_STATUS = "error"
        JULIA_INIT_ERROR_MSG = f"Failed to read worker handshake: {exc}"
        print(JULIA_INIT_ERROR_MSG)
        return

    if msg.get("ready"):
        JULIA_STATUS = "ready"
        print("Julia worker is ready.")
    else:
        JULIA_STATUS = "error"
        JULIA_INIT_ERROR_MSG = msg.get("error", "Unknown worker startup error")
        print(f"Julia worker failed: {JULIA_INIT_ERROR_MSG}")

async def trigger_julia_init():
    """Triggers Julia worker startup if not yet started (non-blocking after first call)."""
    global JULIA_STATUS
    if JULIA_STATUS == "idle":
        # Fire-and-forget: start worker in background without holding the lock
        asyncio.create_task(_start_julia_worker_bg())

async def _start_julia_worker_bg():
    """Background task wrapper that acquires the lock before starting the worker."""
    async with _julia_lock:
        await _start_julia_worker()

async def _send_julia_request(payload: dict) -> dict:
    """Send a JSON request to the Julia worker and return the parsed JSON response."""
    global _julia_proc
    if _julia_proc is None or _julia_proc.returncode is not None:
        raise RuntimeError("Julia worker process is not running.")
    line = json.dumps(payload) + "\n"
    _julia_proc.stdin.write(line.encode())
    await _julia_proc.stdin.drain()
    raw = await _julia_proc.stdout.readline()
    if not raw:
        stderr_out = await _julia_proc.stderr.read()
        raise RuntimeError(
            f"Julia worker closed unexpectedly. stderr: {stderr_out.decode()[:500]}"
        )
    return json.loads(raw.decode().strip())

# --- Routes ---

@app.on_event("startup")
async def on_startup():
    """Eagerly start the Julia worker in the background so it's warm when first needed."""
    asyncio.create_task(_start_julia_worker_bg())

@app.get("/")
def get_gui():
    """Serves the static index.html user interface."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html templates file not found.")
    return FileResponse(html_path)

@app.get("/api/components")
def get_components_metadata():
    """Returns available component classes, their inputs, and descriptors."""
    return COMPONENTS

@app.get("/api/julia/status")
def get_julia_status():
    """Returns the current Julia initialization status so the frontend can poll readiness."""
    return {
        "status": JULIA_STATUS,
        "error": JULIA_INIT_ERROR_MSG,
        "cuda_available": False  # placeholder; updated after init
    }

@app.post("/api/julia/init")
async def init_julia_backend():
    """Triggers Julia initialization if it hasn't started yet. Returns current status."""
    if JULIA_STATUS == "idle":
        await trigger_julia_init()
    return {"status": JULIA_STATUS, "error": JULIA_INIT_ERROR_MSG}


@app.post("/api/compile")
def compile_graph(req: CompileRequest):
    """Compiles the graphical components and node wires into an index-reduced explicit ODE."""
    if not req.nodes:
        raise HTTPException(status_code=400, detail="Cannot compile an empty system. Place some components first.")
        
    comp_classes = get_component_classes()
    
    try:
        # 1. Instantiate Braid Component classes
        instances = {}
        for n in req.nodes:
            if n.type not in comp_classes:
                raise HTTPException(status_code=400, detail=f"Unsupported component type: {n.type}")
            # Dynamic instantiation, passing name and parameter overrides
            instances[n.id] = comp_classes[n.type](name=n.id, **n.params)
            
        # 2. Build Braid System
        system = System(list(instances.values()))
        
        # 3. Connect ports by groupings from graph edges
        groups = build_connected_groups(req.nodes, req.edges)
        for group in groups:
            if len(group) > 1:
                comp_set = [(instances[comp_id], port_name) for comp_id, port_name in group]
                Node(system, comp_set)
                
        # 4. Perform SymPy DAE Assembly & Compiler Passes
        dae = system.to_dae()
        initial_eqs = [str(eq) for eq in dae.equations]
        
        # Index Reduction Pass (Pantelides)
        reduced_dae = pantelides_pass(dae)
        reduced_eqs = [str(eq) for eq in reduced_dae.equations]
        
        # Tearing Pass (Symbolic Solver)
        torn_dae = tearing_pass(reduced_dae)
        torn_eqs = [str(eq) for eq in torn_dae.equations]
        
        # Elimination & Simplification Pass
        simplified_dae = simplification_pass(torn_dae)
        
        # Verify that all state derivatives are defined in the ODE system
        import sympy as sp
        missing_derivs = []
        for state in simplified_dae.states:
            state_deriv = sp.Derivative(state, dae.t)
            if state_deriv not in simplified_dae.ode_assignments:
                missing_derivs.append(str(state))
                
        if missing_derivs:
            raise ValueError(
                f"Structurally singular system. The derivatives of the following states are under-constrained: {', '.join(missing_derivs)}. "
                "Ensure all components (especially masses) are connected correctly and there are no disconnected nodes."
            )
            
        ode_assignments = {str(k): str(v) for k, v in simplified_dae.ode_assignments.items()}
        
        # Save compiled model in memory
        compile_id = str(uuid.uuid4())
        COMPILED_MODELS[compile_id] = simplified_dae
        
        return {
            "compile_id": compile_id,
            "states": [str(s) for s in simplified_dae.states],
            "params": [str(p) for p in simplified_dae.params],
            "initial_eqs": initial_eqs,
            "reduced_eqs": reduced_eqs,
            "torn_eqs": torn_eqs,
            "ode_assignments": ode_assignments,
            "components": [{"name": n.id, "type": n.type} for n in req.nodes],
            "sensor_targets": {k: str(v) for k, v in getattr(simplified_dae, 'sensor_targets', {}).items()}
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Compilation failed: {str(e)}")

@app.post("/api/simulate")
async def run_simulation(req: SimulateRequest):
    """Executes the numerical integration solver on the compiled system."""
    simp_dae = COMPILED_MODELS.get(req.compile_id)
    if not simp_dae:
        raise HTTPException(status_code=404, detail="Model compile state not found. Please compile the system first.")

    # Guard: if Julia is requested but not yet ready, return a descriptive error
    if req.backend.lower() == "julia":
        if JULIA_STATUS == "initializing":
            raise HTTPException(
                status_code=503,
                detail="Julia is still initializing. This can take 1–3 minutes on first use. "
                       "Check the Julia Status indicator and try again shortly."
            )
        elif JULIA_STATUS == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Julia initialization failed: {JULIA_INIT_ERROR_MSG}"
            )
        elif JULIA_STATUS != "ready":
            # Unexpected state — trigger init and tell user to wait
            await trigger_julia_init()
            raise HTTPException(
                status_code=503,
                detail="Julia initialization has not started. Triggering now — please wait a moment and try again."
            )

    # --- Build common inputs ---
    y0_list = []
    for state in simp_dae.states:
        val = req.y0.get(str(state), 0.0)
        y0_list.append(val)

    params_dict = dict(req.params)

    def _build_response(sol, params_dict):
        """Convert a simulation result into the JSON response dict."""
        import numpy as np
        import sympy as sp

        response_data = {
            "t": sol.t.tolist() if hasattr(sol.t, "tolist") else list(sol.t),
            "states": {}
        }

        sol_y = sol.y
        # Ensure 2D: (n_states, n_steps)
        if isinstance(sol_y, list):
            sol_y = np.array(sol_y)
        if sol_y.ndim == 1:
            sol_y = sol_y.reshape(1, -1)

        for idx, state in enumerate(simp_dae.states):
            values = sol_y[idx]
            response_data["states"][str(state)] = values.tolist() if hasattr(values, "tolist") else list(values)

        # Evaluate solved assignments (algebraic variables, sensors, etc.)
        if simp_dae.solved_assignments:
            flat_params = []
            param_meta = getattr(simp_dae, "param_meta", {})
            for p in simp_dae.params:
                val = None
                found = False
                if p in params_dict:
                    val = params_dict[p]
                    found = True
                elif p.name in params_dict:
                    val = params_dict[p.name]
                    found = True
                elif param_meta:
                    sym_repr = sp.srepr(p)
                    if sym_repr in param_meta:
                        meta = param_meta[sym_repr]
                        comp_dot_name = f"{meta['component']}.{meta['name']}"
                        if comp_dot_name in params_dict:
                            val = params_dict[comp_dot_name]
                            found = True
                        elif 'default' in meta:
                            val = meta['default']
                            found = True
                if not found:
                    val = 0.0
                flat_params.append(val)

            vars_list = list(simp_dae.solved_assignments.keys())
            exprs_list = list(simp_dae.solved_assignments.values())
            eval_fn = sp.lambdify((simp_dae.t, simp_dae.states, simp_dae.params), exprs_list, 'numpy')

            solved_values = {str(var): [] for var in vars_list}
            for i in range(len(response_data["t"])):
                t_val = response_data["t"][i]
                y_val = sol_y[:, i]
                res = eval_fn(t_val, y_val, flat_params)
                for var_idx, var in enumerate(vars_list):
                    solved_values[str(var)].append(float(res[var_idx]))

            for var_str, vals in solved_values.items():
                response_data["states"][var_str] = vals

        return response_data

    try:
        if req.backend.lower() == "julia":
            # --- Julia path: communicate with the worker subprocess ---
            # Resolve params dict → flat list (same logic as simulate_system)
            import sympy as sp
            import numpy as np
            param_meta = getattr(simp_dae, "param_meta", {})
            flat_params = []
            for p in simp_dae.params:
                val = None
                found = False
                if p in params_dict:
                    val = float(params_dict[p]); found = True
                elif p.name in params_dict:
                    val = float(params_dict[p.name]); found = True
                elif param_meta:
                    sym_repr = sp.srepr(p)
                    if sym_repr in param_meta:
                        meta = param_meta[sym_repr]
                        comp_dot_name = f"{meta['component']}.{meta['name']}"
                        if comp_dot_name in params_dict:
                            val = float(params_dict[comp_dot_name]); found = True
                        elif 'default' in meta:
                            val = float(meta['default']); found = True
                if not found:
                    val = 0.0
                flat_params.append(val)

            dae_json_str = dae_to_json(simp_dae)

            async with _julia_lock:
                resp = await _send_julia_request({
                    "type": "simulate",
                    "dae_json": dae_json_str,
                    "t_span": list(req.t_span),
                    "y0": [float(v) for v in y0_list],
                    "params": flat_params,
                    "method": req.method,
                    "device": None,
                    "num_steps": req.num_steps,
                })

            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "Julia worker returned an error."))

            import numpy as np

            class _JuliaResult:
                def __init__(self, t, y):
                    self.t = np.array(t)
                    self.y = np.array(y)  # shape (n_states, n_steps)
                    self.success = True

            sol = _JuliaResult(resp["t"], resp["y"])
            return _build_response(sol, params_dict)

        else:
            # --- Non-Julia path: run directly in the asyncio thread ---
            def _do_simulate():
                sol = simulate_system(
                    dae=simp_dae,
                    t_span=tuple(req.t_span),
                    y0=y0_list,
                    params=params_dict,
                    backend=req.backend,
                    method=req.method,
                    num_steps=req.num_steps
                )
                if not sol.success:
                    raise RuntimeError(
                        f"Solver integration failed: {getattr(sol, 'message', 'unknown solver error')}"
                    )
                return sol

            loop = asyncio.get_event_loop()
            sol = await loop.run_in_executor(None, _do_simulate)
            return _build_response(sol, params_dict)

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Start FastAPI server on localhost:8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
