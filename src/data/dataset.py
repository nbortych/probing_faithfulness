from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


@dataclass
class ModelResponse:
    """Single model response with metadata."""
    id: str  # Unique identifier
    prompt: str
    response: str
    is_faithful: bool
    model_name: str
    faithfulness_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class FaithfulnessDataset:
    """Dataset for faithful/unfaithful model responses."""
    
    def __init__(self, save_dir: Path):
        """Initialize dataset.
        
        Args:
            save_dir: Directory for saving dataset files
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.responses: List[ModelResponse] = []
        
    def add_response(self, 
                    id: str,
                    prompt: str, 
                    response: str, 
                    is_faithful: bool,
                    model_name: str,
                    faithfulness_type: Optional[str] = None, 
                    metadata: Optional[Dict[str, Any]] = None):
        """Add a single response to the dataset."""
        response = ModelResponse(
            id=id,
            prompt=prompt,
            response=response,
            is_faithful=is_faithful,
            model_name=model_name,
            faithfulness_type=faithfulness_type,
            metadata=metadata or {}
        )
        self.responses.append(response)
        
    def save(self, filename: str = "faithfulness_dataset.json"):
        """Save the dataset to a JSON file."""
        path = self.save_dir / filename
        data = [r.to_dict() for r in self.responses]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: Path) -> 'FaithfulnessDataset':
        """Load a dataset from a JSON file."""
        dataset = cls(save_dir=path.parent)
        with open(path) as f:
            data = json.load(f)
        dataset.responses = [ModelResponse(**r) for r in data]
        return dataset
    
    def get_faithful_responses(self) -> List[ModelResponse]:
        """Get all faithful responses."""
        return [r for r in self.responses if r.is_faithful]
    
    def get_unfaithful_responses(self) -> List[ModelResponse]:
        """Get all unfaithful responses."""
        return [r for r in self.responses if not r.is_faithful]
    
    def __len__(self) -> int:
        return len(self.responses)
    
    def __getitem__(self, idx: int) -> ModelResponse:
        return self.responses[idx]