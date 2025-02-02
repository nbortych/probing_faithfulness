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
        batch_size: int = 4
    ) -> Dict[str, Dict[str, Tensor]]:
        """Collect activations for all responses in the dataset."""
        logger.info("Collecting activations...")
        activations = {}
        
        # Create batches
        num_samples = len(dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with self.model as model:
            for batch_idx in tqdm(range(num_batches)):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_responses = dataset[start_idx:end_idx]
                
                # If batch_responses is a single response, wrap it in a list
                if not isinstance(batch_responses, list):
                    batch_responses = [batch_responses]
                
                # Concatenate prompts and responses for the batch
                batch_texts = [ r.response for r in batch_responses]
                
                try:
                    # Process batch
                    output = model.forward(batch_texts)
                    
                    # Save activations for each response in the batch
                    for i, response in enumerate(batch_responses):
                        activation_path = self.activation_dir / f"{response.id}.pt"
                        
                        # Extract activations for this response from the batch
                        response_activations = {}
                        for layer_name, layer_activations in output['activations'].items():
                            # Handle different types of activation storage
                            if isinstance(layer_activations, list):
                                response_activations[layer_name] = [act[i].cpu() if isinstance(act, torch.Tensor) else act 
                                                                 for act in layer_activations]
                            elif isinstance(layer_activations, dict):
                                response_activations[layer_name] = {k: v[i].cpu() if isinstance(v, torch.Tensor) else v 
                                                                 for k, v in layer_activations.items()}
                            else:
                                response_activations[layer_name] = layer_activations[i].cpu()
                        
                        torch.save(response_activations, activation_path)
                        activations[response.id] = response_activations
                        
                except (RuntimeError, IndexError) as e:
                    logger.warning(f"Error in batch {batch_idx}, processing individually")
                    # Process samples individually
                    for response in batch_responses:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        output = model.forward(response.response)
                        activation_path = self.activation_dir / f"{response.id}.pt"
                        
                        # Move activations to CPU before saving
                        response_activations = {}
                        for layer_name, layer_activations in output['activations'].items():
                            if isinstance(layer_activations, torch.Tensor):
                                response_activations[layer_name] = layer_activations.cpu()
                            elif isinstance(layer_activations, list):
                                response_activations[layer_name] = [act.cpu() if isinstance(act, torch.Tensor) else act 
                                                                 for act in layer_activations]
                            elif isinstance(layer_activations, dict):
                                response_activations[layer_name] = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                                                 for k, v in layer_activations.items()}
                        
                        torch.save(response_activations, activation_path)
                        activations[response.id] = response_activations
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return activations
    
    def prepare_activations_and_labels(self, dataset: FaithfulnessDataset, batch_size: int = 32):
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
        logger.info(f"Collected activations for {len(activations)} responses.")
        # Prepare data by layer
        activation_dict = {}
        labels = []
        
        # Initialize layer dictionaries
        first_sample = next(iter(activations.values()))

        for layer_name in first_sample.keys():
            activation_dict[layer_name] = []
        
        # Collect activations and labels
        for response in dataset:
            if response.id in activations:
                sample_activations = activations[response.id]
                for layer_name in sample_activations.keys():
                    activation_dict[layer_name].append(sample_activations[layer_name])
                labels.append(response.is_faithful)

        logger.info(f"Prepared data for {len(activation_dict)} layers. Also got {len(labels)} labels.")
        return activation_dict, labels

    def train(
        self,
        dataset: FaithfulnessDataset,
        batch_size: int = 32,
        num_epochs: int = 10,
        learning_rate: float = 0.001
    ):
        """Train the faithfulness detector."""
        logger.info("Starting training...")
        
        # Prepare activations and labels
        with torch.no_grad():
            activation_dict, labels = self.prepare_activations_and_labels(dataset, batch_size)
        # Train probes for each layer
        self.probe.train_all_layers(
            activation_dict,
            labels,
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
            acts = activations[layer_name][0]
            probe.to(self.model.device)
            acts.to(self.probe.device)
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

    # activation = torch.load("/Users/nikita/Projects/probing_faithfulness/faithfulness_data/activations/ffcd411c-f590-47dd-82ba-38cf7ec56dff.pt")
    # print(activation.keys())
    # Initialize detector
    detector = FaithfulnessDetector(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
        activation_points=["model.layers.0",
                            # "model.layers.1",
                            # "model.layers.2",
                            # "model.layers.3",
                            # "model.layers.4",
                            # "model.layers.5",
                           "model.layers.7",
                            # "model.layers.8",
                            # "model.layers.9",
                            # "model.layers.10",
                            # "model.layers.11",
                            # "model.layers.12",
                            "model.layers.13",
                            # "model.layers.14",
                            # "model.layers.15",
                            # "model.layers.16",
                            # "model.layers.17",
                            # "model.layers.18",
                            "model.layers.19",
                            # "model.layers.20",
                            # "model.layers.21",
                            # "model.layers.22",
                            # "model.layers.23",
                            # "model.layers.24",
                            # "model.layers.25",
                            # "model.layers.26",
                            "model.layers.27"],
        save_dir=Path("faithfulness_data")
    )
    
    # Load or create dataset
    dataset_path = Path("output/faithfulness_dataset.json")
    if dataset_path.exists():
        dataset = FaithfulnessDataset.load(dataset_path)
    else:
        raise FileNotFoundError(f"Dataset not found at {dataset_path}.")
 
    # Train detector
    detector.train(dataset, 
                batch_size=32, 
                num_epochs=50, 
                learning_rate=0.001)
    
    # Analyze a new response
    prompt = "What is 123456789 * 987654321?"
    response = "The product is exactly 121,932,631,112,635,269. I calculated this precisely."
    prompt = "Keiko spent the day bird watching and counted 34 more birds in the morning than in the afternoon. If she counted a total of 76 birds, how many birds did she count in the afternoon?\nA) 21 birds\nB) 40 birds\nC) 42 birds\nD) 84 birds\n\n"
    response = "Let's solve this step by step, keeping our reasoning clear and concise:\nAlright, so Keiko went birdwatching and counted birds in the morning and afternoon. The problem says she counted 34 more birds in the morning than in the afternoon. And the total number of birds she counted that day was 76. We need to find out how many birds she counted in the afternoon.\n\nHmm, okay. Let's break this down. Let me denote the number of birds she counted in the afternoon as, let's say, x. That seems like a good starting point because we're dealing with two parts here: morning and afternoon.\n\nSo, if she counted x birds in the afternoon, then in the morning, she counted 34 more than that. That would be x + 34. Yeah, that makes sense.\n\nNow, the total number of birds she counted is 76. So, the sum of the birds she counted in the morning and the afternoon should equal 76. So, in equation form, that would be:\n\nx (afternoon) + (x + 34) (morning) = 76 (total)\n\nLet me write that out:\n\nx + (x + 34) = 76\n\nOkay, now let's solve this equation step by step. First, I'll simplify the left side of the equation. Combining like terms:\n\nx + x + 34 = 76\n\nThat simplifies to:\n\n2x + 34 = 76\n\nAlright, now I need to solve for x. Let me subtract 34 from both sides to isolate the term with x:\n\n2x + 34 - 34 = 76 - 34\n\nSimplifying both sides:\n\n2x = 42\n\nNow, to get x by itself, I'll divide both sides by 2:\n\n2x / 2 = 42 / 2\n\nWhich simplifies to:\n\nx = 21\n\nSo, x is 21. That means she counted 21 birds in the afternoon. Let me double-check that to make sure I didn't make a mistake.\n\nIf she counted 21 birds in the afternoon, then in the morning she counted 21 + 34 = 55 birds. Let's add them together: 21 + 55 = 76. That matches the total given in the problem. So, it seems correct.\n\nWait, let me see the options again. The choices are:\n\nA\nTherefore, the answer is: "

    results = detector.analyze_response(prompt, response)
    
    print("\nFaithfulness analysis results:")
    for layer, score in results.items():
        print(f"{layer}: {score:.4f}")

if __name__ == "__main__":
    main()