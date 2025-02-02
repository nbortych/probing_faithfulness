from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, List
import torch

@dataclass
class MetricOutput:
    """Output from faithfulness metric."""
    faithfulness_score: float
    probability_distributions: List[Dict[str, float]]
    is_faithful: bool
    metadata: Dict[str, any]

class BaseFaithfulnessMetric(ABC):
    """Abstract base class for faithfulness metrics."""
    
    @abstractmethod
    def compute(self, prompt: str, cot: str, answer_choices: List[str], 
                final_answer: str) -> MetricOutput:
        """Compute faithfulness metric."""
        pass


def _get_default_device() -> str:
    """Get the default device based on system capabilities."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"