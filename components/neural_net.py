"""
components/neural_net.py
========================
Component that imports a PyTorch neural network and converts it to CasADi expressions.
"""

import casadi as ca
import torch
import torch.nn as nn
from base import Component

def pytorch_to_casadi(model: nn.Module, x_in: ca.SX) -> ca.SX:
    """
    Translates a PyTorch neural network forward pass into a CasADi SX expression.
    Supports Sequential, Linear, ReLU, Sigmoid, and Tanh layers.
    """
    current_expr = x_in
    
    # helper function to handle linear layer
    def handle_linear(layer: nn.Linear, expr: ca.SX) -> ca.SX:
        weight = layer.weight.detach().cpu().numpy()
        W = ca.SX(weight)
        if layer.bias is not None:
            bias = layer.bias.detach().cpu().numpy()
            b = ca.SX(bias)
            return ca.mtimes(W, expr) + b
        else:
            return ca.mtimes(W, expr)

    # helper function to traverse modules
    def traverse(module: nn.Module, expr: ca.SX) -> ca.SX:
        if isinstance(module, nn.Sequential):
            for child in module.children():
                expr = traverse(child, expr)
            return expr
        elif isinstance(module, nn.Linear):
            return handle_linear(module, expr)
        elif isinstance(module, nn.ReLU):
            return ca.fmax(0.0, expr)
        elif isinstance(module, nn.Sigmoid):
            return 1.0 / (1.0 + ca.exp(-expr))
        elif isinstance(module, nn.Tanh):
            return (ca.exp(expr) - ca.exp(-expr)) / (ca.exp(expr) + ca.exp(-expr))
        else:
            raise NotImplementedError(
                f"Translation of PyTorch layer type {type(module)} is not supported."
            )

    return traverse(model, current_expr)


class NeuralNetworkPyTorch(Component):
    """
    Component wrapping a PyTorch neural network.
    Reconstructs the PyTorch forward pass as CasADi expressions.
    """
    def __init__(self, name: str, pytorch_model: nn.Module, input_names: list, output_names: list):
        super().__init__(name)
        
        self.pytorch_model = pytorch_model
        self.input_names = input_names
        self.output_names = output_names
        
        # 1. Register input ports and create internal input symbols
        self.input_syms = []
        for inp_name in input_names:
            sym = ca.SX.sym(f"in_{self.name}_{inp_name}")
            self.input_syms.append(sym)
            # Ports: effort=0, across=sym, dacross=0
            self.ports[inp_name] = [ca.SX(0), sym, ca.SX(0)]
            
        # Stack inputs into a vector for matrix operations
        x_in = ca.vertcat(*self.input_syms)
        
        # 2. Translate PyTorch model to CasADi expression
        self.nn_expr = pytorch_to_casadi(pytorch_model, x_in)
        
        # 3. Create output symbols, add DAE equations, and register output ports
        self.output_syms = []
        for i, out_name in enumerate(output_names):
            sym = ca.SX.sym(f"out_{self.name}_{out_name}")
            self.output_syms.append(sym)
            
            # Constraint equation: output_sym - nn_expr[i] = 0
            self.equations.append(sym - self.nn_expr[i])
            
            # Ports: effort=sym, across=0, dacross=0
            self.ports[out_name] = [sym, ca.SX(0), ca.SX(0)]
