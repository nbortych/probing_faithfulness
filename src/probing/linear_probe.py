from typing import Dict, List, Tuple, Optional
import logging
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

logger = logging.getLogger(__name__)
class ActivationDataset(Dataset):
    """Dataset for storing model activations with faithfulness labels."""
    
    def __init__(self, activations: Tensor, labels: List[bool]):
        """Initialize the dataset.
        
        Args:
            activations: List of activation tensors
            labels: List of faithfulness labels (True for faithful, False for unfaithful)
        """
        self.activations = activations
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.activations[idx], self.labels[idx]

class LinearProbe(nn.Module):
    """Linear probe for detecting faithfulness from activations."""
    
    def __init__(self, input_dim: int):
        """Initialize the probe.
        
        Args:
            input_dim: Dimension of input activations
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the probe."""
        return self.sigmoid(self.linear(x))

class FaithfulnessProbe:
    """Manager class for training and using linear probes."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the probe manager.
        
        Args:
            device: Device to run on (default: auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.probes: Dict[str, LinearProbe] = {}
        self.best_layer: Optional[str] = None
        self.best_accuracy: float = 0.0
        

    def pool_sequence_activations(
        self,
        activations: torch.Tensor,
        method: str = "mean"
    ) -> torch.Tensor:
        """
        Pool transformer activations along the sequence dimension.
        
        Args:
            activations: Tensor of shape (seq_len, hidden_dim)
            method: Pooling method - "mean" or "last"
        
        Returns:
            Pooled tensor of shape (hidden_dim,)
        """
        if method == "mean":
            return activations.mean(dim=0)  # Average across sequence length
        elif method == "last":
            return activations[-1, :]       # Take last token's activations
        else:
            raise ValueError(f"Invalid pooling method: {method}. Choose 'mean' or 'last'.")
        
    
    def prepare_activation_data(self, 
                          activation_dict: Dict[str, Dict[str, Tensor]],
                          labels: List[bool], pooling_method: str = "last") -> Dict[str, ActivationDataset]:
        """Prepare activation data for each layer."""
        datasets = {}
        for layer_name, layer_acts in activation_dict.items():
            acts_list = []
            for act in layer_acts:
                if isinstance(act, torch.Tensor):
                    pooled = self.pool_sequence_activations(act, method=pooling_method)
                elif isinstance(act, (list,dict)):
                    if isinstance(act, dict):
                        act = list(act.values())
                    tensors = [t for t in act if isinstance(t, torch.Tensor)]
                    if not tensors:
                        continue
                    concatenated = torch.cat(tensors, dim=-1)
                    pooled = self.pool_sequence_activations(concatenated, method=pooling_method)
                else:
                    logger.info(f"Skipping activation of type {type(act)}")
                    continue
                acts_list.append(pooled)
            
            if not acts_list:
                logger.info(f"Warning: No valid tensors found for layer {layer_name}")
                continue
                
            # Stack tensors along batch dimension
            if acts_list:
                try:
                    acts = torch.stack(acts_list)
                        
                    datasets[layer_name] = ActivationDataset(acts, labels)
                except Exception as e:
                    logger.info(f"Error processing layer {layer_name}: {str(e)}")
                    logger.info(f"Shapes of tensors in acts_list: {[a.shape for a in acts_list]}")
                    raise
                    
        if not datasets:
            raise ValueError("No valid datasets created from activations")
            
        return datasets
    
    def train_probe(self, 
                   dataset: ActivationDataset,
                   batch_size: int = 32,
                   num_epochs: int = 10,
                   learning_rate: float = 0.001) -> Tuple[LinearProbe, float]:
        """Train a linear probe on the given dataset.
        
        Args:
            dataset: ActivationDataset to train on
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            
        Returns:
            Tuple of (trained probe, validation accuracy)
        """
        # Split data
        train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize probe
        input_dim = dataset[0][0].shape[-1]
        probe = LinearProbe(input_dim).to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_acc = 0.0
        best_state = None
        
        for epoch in range(num_epochs):
            probe.train()
            for acts, labels in train_loader:
                acts, labels = acts.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = probe(acts).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            probe.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for acts, labels in val_loader:
                    acts, labels = acts.to(self.device), labels.to(self.device)
                    outputs = probe(acts).squeeze()
                    predictions = (outputs > 0.5).float()
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = probe.state_dict()
        
        # Load best state
        probe.load_state_dict(best_state)
        return probe, best_val_acc
    
    def train_all_layers(self, 
                        activation_dict: Dict[str, Dict[str, Tensor]],
                        labels: List[bool],
                        **train_kwargs):
        """Train probes for all layers and find the best one.
        
        Args:
            activation_dict: Dictionary of activations for each layer
            labels: Corresponding faithfulness labels
            **train_kwargs: Additional arguments for train_probe
        """
        datasets = self.prepare_activation_data(activation_dict, labels)
        
        for layer_name, dataset in datasets.items():
            probe, accuracy = self.train_probe(dataset, **train_kwargs)
            self.probes[layer_name] = probe
            logger.info(f"Layer {layer_name}: Validation accuracy = {accuracy}")
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_layer = layer_name
    
    def save_probes(self, save_dir: Path):
        """Save trained probes and metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save probes
        for layer_name, probe in self.probes.items():
            torch.save(probe.state_dict(), save_dir / f"{layer_name}_probe.pt")
        
        # Save metadata
        metadata = {
            'best_layer': self.best_layer,
            'best_accuracy': self.best_accuracy
        }
        torch.save(metadata, save_dir / "probe_metadata.pt")
    
    @classmethod
    def load_probes(cls, save_dir: Path) -> 'FaithfulnessProbe':
        """Load saved probes and metadata."""
        save_dir = Path(save_dir)
        probe_manager = cls()
        
        # Load metadata
        metadata = torch.load(save_dir / "probe_metadata.pt")
        probe_manager.best_layer = metadata['best_layer']
        probe_manager.best_accuracy = metadata['best_accuracy']
        
        # Load probes
        for probe_path in save_dir.glob("*_probe.pt"):
            layer_name = probe_path.stem.replace("_probe", "")
            state_dict = torch.load(probe_path)
            input_dim = next(iter(state_dict.values())).shape[1]  # Get input dim from weights
            probe = LinearProbe(input_dim)
            probe.load_state_dict(state_dict)
            probe_manager.probes[layer_name] = probe
        
        return probe_manager