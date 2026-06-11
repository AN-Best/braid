import sys
import os
import uuid
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

# --- Routes ---

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
def run_simulation(req: SimulateRequest):
    """Executes the numerical integration solver on the compiled system."""
    simp_dae = COMPILED_MODELS.get(req.compile_id)
    if not simp_dae:
        raise HTTPException(status_code=404, detail="Model compile state not found. Please compile the system first.")
        
    try:
        # Build y0 list matched to the order of states in simp_dae.states
        y0_list = []
        for state in simp_dae.states:
            state_str = str(state)
            # Find in user y0 overrides, defaulting to 0.0
            val = req.y0.get(state_str, 0.0)
            y0_list.append(val)
            
        # Build parameter overrides dictionary matching Braid dot notation (e.g. mass_1.m)
        # Note: the input params keys are already component_name.param_name, e.g. mass_1.m
        params_dict = {}
        for k, v in req.params.items():
            params_dict[k] = v
            
        # Run Braid simulation engine
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
            raise HTTPException(status_code=500, detail=f"Solver integration failed: {getattr(sol, 'message', 'unknown solver error')}")
            
        # Construct response data
        response_data = {
            "t": sol.t.tolist() if hasattr(sol.t, "tolist") else list(sol.t),
            "states": {}
        }
        
        for idx, state in enumerate(simp_dae.states):
            state_str = str(state)
            values = sol.y[idx]
            response_data["states"][state_str] = values.tolist() if hasattr(values, "tolist") else list(values)
            
        # Evaluate solved assignments (algebraic variables, sensors, etc.)
        if simp_dae.solved_assignments:
            import sympy as sp
            # Get flat parameter values matched to simp_dae.params
            flat_params = []
            param_meta = getattr(simp_dae, "param_meta", {})
            for p in simp_dae.params:
                val = None
                found = False
                
                # Match by Symbol object key or name string key
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
                
            # Lambdify solved assignments using numpy
            vars_list = list(simp_dae.solved_assignments.keys())
            exprs_list = list(simp_dae.solved_assignments.values())
            eval_fn = sp.lambdify((simp_dae.t, simp_dae.states, simp_dae.params), exprs_list, 'numpy')
            
            num_steps_actual = len(response_data["t"])
            solved_values = {str(var): [] for var in vars_list}
            
            for i in range(num_steps_actual):
                t_val = response_data["t"][i]
                y_val = sol.y[:, i] if hasattr(sol.y, "shape") else [sol.y[j][i] for j in range(len(simp_dae.states))]
                
                res = eval_fn(t_val, y_val, flat_params)
                for var_idx, var in enumerate(vars_list):
                    solved_values[str(var)].append(float(res[var_idx]))
                    
            for var_str, vals in solved_values.items():
                response_data["states"][var_str] = vals
                
        return response_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Start FastAPI server on localhost:8000
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
