"""
CUDA Open Compiler - Intermediate Representation (OpenIR)

Inspiré de MLIR, OpenIR est une représentation flexible et extensible du programme.
C'est ici que le code Python est transformé en une structure manipulable par nos optimiseurs.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

@dataclass
class Type:
    """Type de base (Tensor, Index, etc.)"""
    name: str
    shape: Optional[tuple] = None
    dtype: Optional[str] = None

    def __str__(self):
        if self.shape:
            return f"tensor<{'x'.join(map(str, self.shape))}x{self.dtype}>"
        return self.name

@dataclass
class Value:
    """Une valeur dans le graphe (résultat d'une opération)"""
    name: str
    type: Type
    defining_op: 'Op' = None
    uses: List['Op'] = field(default_factory=list)

    def __str__(self):
        return f"%{self.name}: {self.type}"

@dataclass
class Op:
    """Une opération dans le graphe IR"""
    name: str  # e.g., "open.cuda.matmul", "open.turbo.quantize"
    operands: List[Value] = field(default_factory=list)
    results: List[Value] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    block: 'Block' = None

    def __post_init__(self):
        for res in self.results:
            res.defining_op = self

    def __str__(self):
        res_str = ", ".join(str(r) for r in self.results)
        ops_str = ", ".join(str(o) for o in self.operands)
        attrs_str = " ".join(f"{k}={v}" for k, v in self.attributes.items())
        return f"{res_str} = {self.name}({ops_str}) {attrs_str}"

@dataclass
class Block:
    """Un bloc d'opérations (comme un basic block)"""
    operations: List[Op] = field(default_factory=list)
    parent: 'Region' = None

    def add_op(self, op: Op):
        op.block = self
        self.operations.append(op)

@dataclass
class Region:
    """Une région contient des blocs (pour le contrôle de flux)"""
    blocks: List[Block] = field(default_factory=list)
    parent_op: Op = None

@dataclass
class Module:
    """Le module racine contenant tout le programme"""
    name: str
    region: Region = field(default_factory=Region)

    def dump(self):
        print(f"module @{self.name} {{")
        for block in self.region.blocks:
            for op in block.operations:
                print(f"  {op}")
        print("}")
