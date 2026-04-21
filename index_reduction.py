from IR_BASE import *


def rewrite(node:Node,dae:DAE):
    new_children = [rewrite(child, dae) for child in node.children()]
    node = node.with_children(new_children) 
    if isinstance(node,Der) and isinstance(node.children()[0],Der):
        new_state = State(f"{node.children()[0].children()[0].name}_dot")
        dae.derivatives[node.children()[0].children()[0].name] = new_state
        dae.equations.append(Subtract(node.children()[0], new_state))
        dae.states.append(new_state)
        node = Der(new_state)
        return node
    else:
        return node
        
def order_reduction_pass(dae:DAE) -> DAE:

    new_dae = DAE()
    new_dae.states = dae.states
    new_dae.derivatives = dae.derivatives

    for eq in dae.equations:
        new_eq = rewrite(eq, new_dae)
        new_dae.equations.append(new_eq)
    return new_dae