from typing import Dict, List, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from models.base_model import BaseModel
from models.activation_hooks import ActivationHook

class TransformersModel(BaseModel):
    """Wrapper for Hugging Face Transformers models with activation collection."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        activation_points: Optional[List[str]] = None
    ):
        """Initialize the model wrapper.
        
        Args:
            model_name: Name of the Hugging Face model to load
            device: Device to run model on (default: auto-detect)
            activation_points: List of layer names to collect activations from
        """
        super().__init__(device)
        
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        device = device or self._get_default_device()
        self.model.to(self.device)
        self.model.eval()
        
        # Set up activation collection
        self.activation_points = activation_points
        if activation_points:
            self._setup_activation_hooks(activation_points)
    
    def _setup_activation_hooks(self, activation_points: List[str]):
        """Set up hooks for collecting activations at specified points.
        
        Args:
            activation_points: List of model layer names to collect from
        """
        for point in activation_points:
            if '.' in point:
                # Handle nested attributes
                module = self.model
                for attr in point.split('.')[:-1]:
                    module = getattr(module, attr)
                layer = getattr(module, point.split('.')[-1])
            else:
                layer = getattr(self.model, point)
                
            hook = ActivationHook(layer, point, self.activation_storage)
            self.hooks.append(hook)

    def __enter__(self):
        """Context manager entry - ensure hooks are set up"""
        if self.activation_points:
            self._setup_activation_hooks(self.activation_points)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.remove_hooks()
        self.clear_activations()
        
    def forward(self, input_text: str) -> Dict[str, Any]:
        """Run forward pass through the model.
        
        Args:
            input_text: Input text to process
            
        Returns:
            Dictionary containing:
                - 'output': Model's output logits
                - 'tokens': Input tokens
                - 'activations': Collected activations (if hooks are set up)
        """
        tokens = self.tokenizer(input_text, return_tensors="pt", padding= True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        return {
            'output': outputs.logits,
            'tokens': tokens,
            'activations': self.activation_storage.copy()
        }
    
    def get_activation_points(self) -> List[str]:
        """Get list of all possible activation collection points."""
        points = []
        
        def _recurse_model(module: torch.nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                points.append(full_name)
                _recurse_model(child, full_name)
        
        _recurse_model(self.model)
        return points
    

if __name__ == "__main__":
    # Test the TransformersModel class
    model = TransformersModel("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    print(model.get_activation_points())
    with model:
        output = model.forward("Hello, world!")
        print(output['output'].shape)
        print(output['activations'].keys())
        # print(output['activations']['transformer.h.0'])
    model.remove_hooks()
    model.clear_activations()
    print(model.activation_storage)
    print(model.hooks)