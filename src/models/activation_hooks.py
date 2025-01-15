from typing import Dict, Any
import torch
from torch import Tensor
from torch.nn import Module

class ActivationHook:
    """Hook for collecting activations from model layers."""
    
    def __init__(self, module: Module, name: str, storage: Dict[str, Any]):
        """Initialize the activation hook.
        
        Args:
            module: PyTorch module to hook into
            name: Identifier for this hook/activation
            storage: Dictionary to store activations in
        """
        self.name = name
        self.storage = storage
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module: Module, input: Tensor, output: Tensor):
        """Forward hook function that stores activations.
        
        Args:
            module: Module being hooked
            input: Input tensor to the module
            output: Output tensor from the module
        """
        # Store activations - handle different output types
        if isinstance(output, (list, tuple)):
            self.storage[self.name] = [o.detach() if isinstance(o, Tensor) else o 
                                     for o in output]
        elif isinstance(output, dict):
            self.storage[self.name] = {k: v.detach() if isinstance(v, Tensor) else v 
                                     for k, v in output.items()}
        else:
            self.storage[self.name] = output.detach()
    
    def remove(self):
        """Remove the hook."""
        self.hook.remove()