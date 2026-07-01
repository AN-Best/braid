"""
Julia worker subprocess.

This script is launched as a child process by app.py when the Julia backend is
requested. It reads newline-delimited JSON requests from stdin and writes
newline-delimited JSON responses to stdout.

Protocol:
  Request JSON: {
    "type": "simulate",
    "dae_json": "<json_ir string>",
    "t_span": [t0, tf],
    "y0": [float, ...],
    "params": [float, ...],
    "method": "rk4",
    "device": null,
    "num_steps": 1000
  }

  Response JSON (success): {"ok": true, "t": [...], "y": [[...], ...]}
  Response JSON (error):   {"ok": false, "error": "<message>"}

A startup handshake is also emitted once Julia is ready:
  {"ready": true}

On failure to initialize Julia:
  {"ready": false, "error": "<message>"}
"""
import sys
import os
import json
import traceback

# Worker must be able to import from project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

def _init():
    """Initialize Julia — called once at startup."""
    try:
        from backends.julia_backend import init_julia
        init_julia()
        _send({"ready": True})
    except Exception as exc:
        _send({"ready": False, "error": str(exc)})
        sys.exit(1)

def _send(obj):
    """Write a JSON line to stdout and flush immediately."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def _handle_simulate(req):
    from json_ir import from_json
    from backends.julia_backend import simulate_julia

    dae = from_json(req["dae_json"])
    t_span = tuple(req["t_span"])
    y0 = req["y0"]
    params = req["params"]
    method = req.get("method")
    device = req.get("device")
    num_steps = req.get("num_steps", 1000)

    result = simulate_julia(dae, t_span, y0, params, method, device,
                             num_steps=num_steps)

    import numpy as np
    t_list = result.t.tolist()
    y_arr = result.y
    # Ensure y is always 2D: (n_states, n_steps)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(1, -1)
    y_list = y_arr.tolist()

    _send({"ok": True, "t": t_list, "y": y_list})

def main():
    # Redirect stderr so it doesn't interfere with the stdout JSON protocol
    _init()

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as e:
            _send({"ok": False, "error": f"JSON parse error: {e}"})
            continue

        req_type = req.get("type")
        try:
            if req_type == "simulate":
                _handle_simulate(req)
            else:
                _send({"ok": False, "error": f"Unknown request type: {req_type!r}"})
        except Exception as exc:
            _send({"ok": False, "error": str(exc) + "\n" + traceback.format_exc()})

if __name__ == "__main__":
    main()
