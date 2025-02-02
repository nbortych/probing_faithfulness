from typing import Dict, List, Optional
import torch
from torch import Tensor
from pathlib import Path
import logging
from tqdm import tqdm


from models.transformers_model import TransformersModel
from data.dataset import FaithfulnessDataset
from probing.linear_probe import FaithfulnessProbe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaithfulnessDetector:
    """Main class for detecting model faithfulness using activation analysis."""
    
    def __init__(
        self,
        model_name: str,
        activation_points: Optional[List[str]] = None,
        device: Optional[str] = None,
        save_dir: Optional[Path] = None
    ):
        """Initialize the detector.
        
        Args:
            model_name: Name of the model to analyze
            activation_points: List of layer names to collect activations from
            device: Device to run on
            save_dir: Directory for saving data and models
        """
        self.model_name = model_name
        self.save_dir = Path(save_dir or "faithfulness_detector")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and probe
        self.model = TransformersModel(
            model_name=model_name,
            activation_points=activation_points,
            device=device
        )
        self.probe = FaithfulnessProbe(device=device)
        
        # Create directories
        self.activation_dir = self.save_dir / "activations"
        self.activation_dir.mkdir(exist_ok=True)
        self.probe_dir = self.save_dir / "probes"
        self.probe_dir.mkdir(exist_ok=True)
    
    def collect_activations(
        self,
        dataset: FaithfulnessDataset,
        batch_size: int = 32
    ) -> Dict[str, Dict[str, Tensor]]:
        """Collect activations for all responses in the dataset."""
        logger.info("Collecting activations...")
        activations = {}
        
        with self.model as model:
            for response in tqdm(dataset):
                output = model.forward(response.prompt + response.response)
                activation_path = self.activation_dir / f"{response.id}.pt"
                torch.save(output['activations'], activation_path)
                activations[response.id] = output['activations']
        
        return activations

    def train(
        self,
        dataset: FaithfulnessDataset,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 0.001
    ):
        """Train the faithfulness detector."""
        logger.info("Starting training...")
        
        # Collect or load activations
        activations = {}
        for response in dataset:
            activation_path = self.activation_dir / f"{response.id}.pt"
            if activation_path.exists():
                activations[response.id] = torch.load(activation_path)
            else:
                logger.info("Collecting new activations...")
                activations = self.collect_activations(dataset, batch_size)
                break
        
        # Prepare data by layer
        layer_data = {}
        for layer_name in next(iter(activations.values())).keys():
            layer_acts = []
            labels = []
            
            for response in dataset:
                if response.id in activations:
                    acts = activations[response.id][layer_name]
                    # Mean pool if needed
                    if len(acts.shape) > 2:
                        acts = acts.mean(dim=1)
                    layer_acts.append(acts)
                    labels.append(response.is_faithful)
            
            layer_data[layer_name] = {
                'activations': torch.cat(layer_acts),
                'labels': torch.tensor(labels, dtype=torch.float32)
            }
        
        # Train probes for each layer
        self.probe.train_all_layers(
            layer_data,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # Save trained probes
        self.probe.save_probes(self.probe_dir)
        
        logger.info(f"Training complete. Best layer: {self.probe.best_layer}")
        logger.info(f"Best accuracy: {self.probe.best_accuracy:.4f}")
    
    def analyze_response(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, float]:
        """Analyze a single response for faithfulness."""
        with self.model as model:
            output = model.forward(prompt + response)
            activations = output['activations']
        
        # Get predictions from each layer's probe
        results = {}
        for layer_name, probe in self.probe.probes.items():
            acts = activations[layer_name]
            if len(acts.shape) > 2:
                acts = acts.mean(dim=1)
            with torch.no_grad():
                score = probe(acts).item()
            results[layer_name] = score
        
        return results
    
    @classmethod
    def load(cls, save_dir: Path, model_name: str) -> 'FaithfulnessDetector':
        """Load a trained detector from disk."""
        detector = cls(model_name=model_name, save_dir=save_dir)
        detector.probe = FaithfulnessProbe.load_probes(save_dir / "probes")
        return detector

def main():
    """Example usage of the FaithfulnessDetector."""
    # Initialize detector
    detector = FaithfulnessDetector(
        model_name="ComCom/gpt2-small",
        activation_points=["transformer.h.0", "transformer.h.5", "transformer.h.11"],
        save_dir=Path("faithfulness_data")
    )
    
    # Load or create dataset
    dataset_path = Path("data/faithfulness_dataset.json")
    if dataset_path.exists():
        dataset = FaithfulnessDataset.load(dataset_path)
    else:
        dataset = FaithfulnessDataset(save_dir=Path("data"))
        # Add examples here
        dataset.save()
    
    # Train detector
    detector.train(dataset)
    
    # Analyze a new response
    prompt = "What is 123456789 * 987654321?"
    response = "The product is exactly 121,932,631,112,635,269. I calculated this precisely."
    results = detector.analyze_response(prompt, response)
    
    print("\nFaithfulness analysis results:")
    for layer, score in results.items():
        print(f"{layer}: {score:.4f}")

if __name__ == "__main__":
    main()