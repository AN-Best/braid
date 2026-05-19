from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def children(self) -> list["Node"]:
        pass

    @abstractmethod
    def with_children(self,new_children: list) -> "Node":
        pass

class Numeric(Node):
    def __init__(self,value:float):
        self.value = value
    def children(self):
        return []
    def with_children(self,new_children: list) -> "Node":
        return Numeric(self.value)
    def __repr__(self):
        return str(self.value)
    

class State(Node):
    def __init__(self,name:str):
        self.name = name
    def children(self):
        return []
    def with_children(self, new_children):
        return State(self.name)
    def __repr__(self):
        return self.name
    

class Parameter(Node):
    def __init__(self,name:str,value:float):
        self.name = name
        self.value = value
    def children(self):
        return []
    def with_children(self, new_children):
        return Parameter(self.name,self.value)
    def __repr__(self):
        return self.name
    
class Time(Node):
    def __init__(self):
        self.time = "t"
    def children(self):
        return []
    def with_children(self, new_children):
        return Time()
    def __repr__(self):
        return self.time
    
class Add(Node):
    def __init__(self,right:Node,left:Node):
        self.right = right
        self.left = left
    def children(self):
        return [self.right, self.left]
    def with_children(self, new_children):
        return Add(new_children[0],new_children[1])
    def __repr__(self):
        return f"({self.right} + {self.left})"
        

class Subtract(Node):
    def __init__(self,right:Node,left:Node):
        self.right = right
        self.left = left
    def children(self):
        return [self.right, self.left]
    def with_children(self, new_children):
        return Subtract(new_children[0],new_children[1])
    def __repr__(self):
        return f"({self.right} - {self.left})"
    

class Multiply(Node):
    def __init__(self,right:Node,left:Node):
        self.right = right
        self.left = left
    def children(self):
        return [self.right, self.left]
    def with_children(self, new_children):
        return Multiply(new_children[0],new_children[1])
    def __repr__(self):
        return f"({self.right} * {self.left})"
    

class Divide(Node):
    def __init__(self,right:Node,left:Node):
        self.right = right
        self.left = left
    def children(self):
        return [self.right, self.left]
    def with_children(self, new_children):
        return Divide(new_children[0],new_children[1])
    def __repr__(self):
        return f"({self.right} / {self.left})"
    

class Der(Node):
    def __init__(self,state:State):
        self.state = state
    def children(self):
        return [self.state]
    def with_children(self, new_children):
        return Der(new_children[0])
    def __repr__(self):
        return f"(Der({self.state}))"


class DAE():
    def __init__(self):
        self.states = []
        self.equations = []
        self.params = []
        self.derivatives = {}

class Model():
    def __init__(self):
        self.components = {}
        self.connections = []

