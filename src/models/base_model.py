from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
from torch import Tensor

class BaseModel(ABC):
    """Abstract base class for all models in the project."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the base model.
        
        Args:
            device: Device to run the model on. If None, will automatically detect.
        """
        self.device = device or self._get_default_device()
        self.hooks = []
        self.activation_storage = {}
    
    @staticmethod
    def _get_default_device() -> str:
        """Get the default device based on system capabilities."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @abstractmethod
    def forward(self, input_text: str) -> Dict[str, Any]:
        """Run forward pass and return outputs with activations.
        
        Args:
            input_text: The input text to process.
            
        Returns:
            Dictionary containing model outputs and collected activations.
        """
        pass
    
    @abstractmethod
    def get_activation_points(self) -> List[str]:
        """Get list of available activation collection points in the model."""
        pass
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activation_storage.clear()

    def remove_hooks(self):
        """Remove all activation collection hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.remove_hooks()
        self.clear_activations()